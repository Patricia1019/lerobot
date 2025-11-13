#!/usr/bin/env python3
"""
Extract camera images from a ROS1 rosbag and save an MP4 video and an actions CSV.

Usage:
    python tools/extract_bag_video.py /path/to/bag.bag --out-video out.mp4 --out-actions actions.csv \
        --image-topic /camera/image_raw --action-topic /joint_states --fps 30

Notes:
- Requires ROS (rosbag, rospy), cv_bridge and OpenCV (cv2).
- This script reads Image messages from the specified image topic and writes them
  to an MP4 file using OpenCV's VideoWriter. It also writes JointState messages
  (or any message on the action topic with a `position`/`data`-like field) to CSV.

"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import rosbag
    import rospy
    # Support both Image and CompressedImage
    from sensor_msgs.msg import Image, JointState, CompressedImage
    from cv_bridge import CvBridge
except Exception as e:
    print("This script requires ROS (rosbag, rospy), sensor_msgs and cv_bridge.\nError:", e)
    sys.exit(1)

import cv2
import numpy as np
try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Extract images and actions from a rosbag into an MP4 and CSV.")
    p.add_argument("bag", help="Path to rosbag file")
    p.add_argument("--out-video", required=False, help="Output MP4 file path (if extracting a single topic). If omitted, files are created per-topic automatically.")
    p.add_argument("--out-actions", required=False, help="Output CSV for actions (optional)")
    p.add_argument(
        "--image-topics",
        required=False,
        help=(
            "Comma-separated list of image topics to extract. "
            "Defaults to the four compressed camera topics used by the robot."
        ),
    )
    p.add_argument("--action-topic", default=None, help="Action topic to extract (JointState) if needed; optional")
    p.add_argument("--fps", type=float, default=15, help="Output video FPS (if omitted, uses average message spacing to preserve original speed)")
    p.add_argument("--encoding", default="bgr8", help="Desired cv_bridge encoding for images (default bgr8)")
    p.add_argument(
        "--rgb-output",
        action="store_false",
        help=(
            "Write output videos in RGB pixel ordering using ffmpeg/imageio. "
            "If not provided (default), OpenCV VideoWriter is used (BGR ordering)."
        ),
    )
    p.add_argument(
        "--swap-channels",
        action="store_false",
        help=(
            "Swap R and B channels before writing. Use this if the recorded bag has channels reversed (yellow appears blue)."
        ),
    )
    args = p.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"Bag file not found: {bag_path}")
        sys.exit(2)

    out_video = Path(args.out_video) if args.out_video else None
    out_actions = Path(args.out_actions) if args.out_actions else None

    # If user requested actions but didn't provide an action topic, default to /joint_states
    action_topic = args.action_topic if args.action_topic is not None else ("/joint_states" if out_actions is not None else None)

    bridge = CvBridge()

    # Default compressed camera topics (match the mapping used by the recorder)
    default_image_topics = [
        "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "/hdas/camera_wrist_left/color/image_raw/compressed",
        "/hdas/camera_wrist_right/color/image_raw/compressed",
    ]

    if args.image_topics:
        image_topics = [t.strip() for t in args.image_topics.split(',') if t.strip()]
    else:
        image_topics = default_image_topics

    # First pass: collect image messages per topic (we'll create one video per topic)
    bag = rosbag.Bag(str(bag_path), "r")
    per_topic_images: dict[str, list[tuple[float, any]]] = {t: [] for t in image_topics}
    per_topic_sizes: dict[str, tuple[int, int]] = {}

    print("Scanning bag for image frames (this may take a while for large bags)...")
    try:
        for topic in image_topics:
            try:
                for _topic, msg, t in bag.read_messages(topics=[topic]):
                    # Convert header stamp to float seconds
                    try:
                        ts = msg.header.stamp.to_sec()
                    except Exception:
                        ts = t.to_sec()

                    cv_img = None
                    # Handle CompressedImage messages
                    if 'CompressedImage' in type(msg).__name__ or hasattr(msg, 'format') and hasattr(msg, 'data'):
                        try:
                            arr = np.frombuffer(msg.data, dtype=np.uint8)
                            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        except Exception as e:
                            print(f"Failed to decode CompressedImage from topic {topic}: {e}")
                            continue
                    else:
                        try:
                            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=args.encoding)
                        except Exception:
                            cv_img = bridge.imgmsg_to_cv2(msg)

                    if cv_img is None:
                        continue

                    per_topic_images[topic].append((ts, cv_img))
                    if topic not in per_topic_sizes:
                        h, w = cv_img.shape[:2]
                        per_topic_sizes[topic] = (w, h)
            except Exception:
                # no messages for this topic — that's fine
                continue
    finally:
        bag.close()

    # Write a video per topic that has frames
    any_frames = False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for topic, frames in per_topic_images.items():
        if not frames:
            print(f"No images found on topic {topic} in {bag_path}")
            continue
        any_frames = True

        timestamps = [ts for ts, _ in frames]
        if args.fps and args.fps > 0:
            fps = args.fps
        else:
            diffs = np.diff(sorted(timestamps))
            mean_dt = float(diffs.mean()) if len(diffs) > 0 else 1.0 / 30.0
            fps = 1.0 / mean_dt

        width, height = per_topic_sizes.get(topic, (frames[0][1].shape[1], frames[0][1].shape[0]))

        # determine output directory: mirror recordings/... -> videos/... when possible
        bag_parts = list(bag_path.resolve().parts)
        out_dir = None
        if 'recordings' in bag_parts:
            idx = bag_parts.index('recordings')
            new_parts = bag_parts[:idx] + ['videos'] + bag_parts[idx+1:-1]
            out_dir = Path(*new_parts)
        else:
            # fallback: put into ./videos/<bag_stem> directory under cwd
            out_dir = Path.cwd() / 'videos' / bag_path.parent.name

        out_dir.mkdir(parents=True, exist_ok=True)

        if out_video is not None:
            out_path = Path(args.out_video)
            if len(image_topics) > 1:
                sanitized = topic.strip('/').replace('/', '_')
                out_path = out_path.with_name(out_path.stem + f"_{sanitized}" + out_path.suffix)
            # if out_video is a directory, place file inside it
            if out_path.is_dir():
                sanitized = topic.strip('/').replace('/', '_')
                out_path = out_path / f"{bag_path.stem}_{sanitized}.mp4"
        else:
            sanitized = topic.strip('/').replace('/', '_')
            out_path = out_dir / f"{bag_path.stem}_{sanitized}.mp4"

        print(f"Writing {len(frames)} frames to {out_path} at {fps:.2f} FPS (frame size {width}x{height})")

        # If user requested RGB output and imageio is available, use ffmpeg via imageio
        if args.rgb_output and _HAS_IMAGEIO:
            try:
                # imageio expects frames in RGB order
                # Use yuv420p pixelformat for wide compatibility with players; ffmpeg will convert RGB->YUV.
                writer = imageio.get_writer(
                    str(out_path), fps=fps, codec='libx264', pixelformat='yuv420p'
                )
                for ts, frame in frames:
                    # Optionally swap channels first (fix bags with channel order reversed)
                    if args.swap_channels:
                        try:
                            frame = frame[..., ::-1]
                        except Exception:
                            pass

                    if frame.ndim == 2:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 4:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    else:
                        # convert BGR (cv2 default) to RGB for correct ordering
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)
                writer.close()
                print(f"Video saved to {out_path} (RGB via imageio)")
            except Exception as e:
                print(f"Failed to write RGB video via imageio: {e}. Falling back to OpenCV writer.")
                sys.exit(1)
                vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                for ts, frame in frames:
                    # Optionally swap channels first (fix bags with channel order reversed)
                    if args.swap_channels:
                        try:
                            frame = frame[..., ::-1]
                        except Exception:
                            pass

                    if frame.ndim == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    else:
                        frame_bgr = frame
                    vw.write(frame_bgr)
                vw.release()
                print(f"Video saved to {out_path} (BGR via OpenCV)")
        else:
            if args.rgb_output and not _HAS_IMAGEIO:
                print("--rgb-output requested but imageio/ffmpeg not available; writing with OpenCV (BGR).")
            vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            for ts, frame in frames:
                if frame.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame_bgr = frame
                vw.write(frame_bgr)
            vw.release()
            print(f"Video saved to {out_path}")

    if not any_frames:
        print("No image frames found for any requested topic(s).")

    # If actions requested, extract JointState messages
    if out_actions is not None and action_topic is not None:
        bag = rosbag.Bag(str(bag_path), "r")
        print(f"Extracting actions from topic {action_topic} into {out_actions}")
        with open(str(out_actions), "w", newline='') as f:
            writer = csv.writer(f)
            # we will write header on first JointState message
            header_written = False
            for topic, msg, t in bag.read_messages(topics=[action_topic]):
                ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()
                # JointState has name and position
                if hasattr(msg, 'name') and hasattr(msg, 'position'):
                    if not header_written:
                        writer.writerow(['timestamp'] + list(msg.name))
                        header_written = True
                    row = [ts] + [float(x) for x in msg.position]
                    writer.writerow(row)
                else:
                    # Try to dump raw fields for unknown message types
                    if not header_written:
                        # create a simple header
                        writer.writerow(['timestamp', 'data'])
                        header_written = True
                    writer.writerow([ts, str(msg)])
        bag.close()
        print(f"Actions saved to {out_actions}")


if __name__ == '__main__':
    main()
