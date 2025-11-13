#!/usr/bin/env bash
# Replay a ROS1 rosbag file with optional topic filtering and playback speed.
# Usage: ./replay_bag.sh /path/to/bag.bag [--rate 1.0] [--topics "/camera/image_raw /joint_states"]

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/bag.bag [--rate <float>] [--topics \"/topic1 /topic2\"]"
  exit 2
fi

BAG_PATH="$1"
shift || true

RATE=1.0
TOPICS=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --rate)
      RATE="$2"
      shift; shift
      ;;
    --topics)
      TOPICS="$2"
      shift; shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

if [ ! -f "$BAG_PATH" ]; then
  echo "Bag file not found: $BAG_PATH"
  exit 2
fi

CMD=(rosbag play "$BAG_PATH" --rate "$RATE" --clock)

if [ -n "$TOPICS" ]; then
  # expand topics into separate arguments
  for t in $TOPICS; do
    CMD+=(--topic "$t")
  done
fi

echo "Running: ${CMD[*]}"
# shellcheck disable=SC2086
exec ${CMD[@]}
