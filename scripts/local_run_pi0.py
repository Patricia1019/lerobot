#!/usr/bin/env python3
"""Run a local inference using the same parameters your robot client uses.

This script mirrors the key flags from the robot client so you can run
the pi0 policy locally without the gRPC server.

Example (using your client values):
  python scripts/local_run.py \
    --robot.type=r1lite_left_arm \
    --robot.id=black \
    --task=dummy \
    --policy_type=pi0 \
    --pretrained_name_or_path=/home/peiqi/codes/lerobot/checkpoints/pi0/1_obj_100/009000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=5 \
    --chunk_size_threshold=0.2 \
    --aggregate_fn_name=weighted_average \
    --state 0.0,0.0,0.0,0.0,0.0,0.0

Notes:
 - The main difficulty reproducing the robot client locally is building a valid
   `lerobot_features` mapping and a `RawObservation` matching what the model
   expects (images, tokenized instructions, state vector shapes). For a quick
   smoke test this script creates a minimal state-only observation. For real
   inference you should re-use the same `lerobot_features` your client used.
"""
import argparse
import json
import time
from pathlib import Path
from pprint import pformat

import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.async_inference.helpers import raw_observation_to_observation, TimedObservation
from lerobot.async_inference.helpers import prepare_image
from lerobot.utils.constants import OBS_STATE


def make_lerobot_features_from_state_dim(dim: int):
    return {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [dim],
            "names": [f"joint{i+1}" for i in range(dim)],
        }
    }


def build_raw_observation_from_state(state_list):
    obs = {f"joint{i+1}": float(v) for i, v in enumerate(state_list)}
    obs["task"] = "dummy"
    return obs


def parse_robot_flag(s: str) -> dict:
    # Accept either `robot.type=r1` or `type=r1` style strings; we just parse into dict
    out = {}
    for item in s.split(','):
        if '=' in item:
            k, v = item.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def main():
    parser = argparse.ArgumentParser()

    # Mirror robot client flags (only the ones relevant for local run)
    parser.add_argument('--robot.type', dest='robot_type', default='r1lite_left_arm')
    parser.add_argument('--robot.id', dest='robot_id', default='black')
    parser.add_argument('--task', default='dummy')
    parser.add_argument('--server_address', default=None, help='ignored for local run')

    parser.add_argument('--policy_type', default='pi0')
    parser.add_argument('--pretrained_name_or_path', required=True)
    parser.add_argument('--policy_device', default='cpu')
    parser.add_argument('--actions_per_chunk', type=int, default=5)
    parser.add_argument('--chunk_size_threshold', type=float, default=0.2)
    parser.add_argument('--aggregate_fn_name', default='weighted_average')

    # Inputs for building observation locally. If none provided, the script will
    # use an embedded default state or the example JSON written to
    # scripts/example_obs.json.
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--state', help='comma-separated state vector, e.g. 0.0,0.1,0.2')
    group.add_argument('--obs_json', help='path to JSON file with a raw observation dict')

    args = parser.parse_args()

    device = args.policy_device
    pretrained = args.pretrained_name_or_path

    print('Configuration:')
    print(pformat(vars(args)))

    # Load policy
    PolicyClass = get_policy_class(args.policy_type)
    print(f'Loading policy {args.policy_type} from {pretrained} ...')
    policy = PolicyClass.from_pretrained(pretrained)
    policy.to(device)

    # Load processors from pretrained (ensures inputs/outputs shaped as the model expects)
    device_override = {'device': device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained,
        preprocessor_overrides={'device_processor': device_override},
        postprocessor_overrides={'device_processor': device_override},
    )

    # Build raw observation. Use defaults if user didn't provide state or JSON.
    default_state_str = "0.0,0.0,0.0,0.0,0.0,0.0"
    example_json_path = Path(__file__).resolve().parent / "example_obs.json"

    if not args.state and not args.obs_json:
        # Prefer state default; also ensure example JSON exists for users who want it
        args.state = default_state_str
        if not example_json_path.exists():
            example_obs = build_raw_observation_from_state([0.0] * 6)
            example_json_path.write_text(json.dumps(example_obs, indent=2))
            print(f"Wrote example observation JSON to {example_json_path}")

    if args.state:
        state_vals = [float(s) for s in args.state.split(',') if s.strip() != '']
        raw_obs = build_raw_observation_from_state(state_vals)
    else:
        raw_path = Path(args.obs_json)
        raw_obs = json.loads(raw_path.read_text())

    # Minimal lerobot_features mapping — for real runs you should use the exact mapping
    # produced by `robot_client` (via map_robot_keys_to_lerobot_features)
    if args.state:
        lerobot_features = make_lerobot_features_from_state_dim(len(state_vals))
    else:
        lerobot_features = make_lerobot_features_from_state_dim(6)

    # Convert raw observation to policy-ready observation and apply preprocessor
    observation = raw_observation_to_observation(raw_obs, lerobot_features, policy.config.image_features)

    processed = preprocessor(observation)

    # Run predict_action_chunk and postprocess actions
    with torch.no_grad():
        actions = policy.predict_action_chunk(processed)[:, : args.actions_per_chunk, :]

    processed_actions = []
    chunk_size = actions.shape[1]
    for i in range(chunk_size):
        single_action = actions[:, i, :]
        processed_action = postprocessor(single_action)
        processed_actions.append(processed_action)

    action_chunk = torch.stack(processed_actions, dim=1).squeeze(0)

    print('Action chunk shape:', action_chunk.shape)
    print(action_chunk)


if __name__ == '__main__':
    main()
