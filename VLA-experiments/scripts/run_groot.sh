python -m lerobot.async_inference.robot_client \
  --robot.type=r1lite   \
  --robot.id=black  \
   --task="dummy"  \
    --server_address=128.2.204.110:8080   \
    --policy_type=groot   \
    --pretrained_name_or_path=/home/peiqi/codes/lerobot/checkpoints/groot/allobj/037000/pretrained_model  \
    --policy_device=cuda   \
    --actions_per_chunk=3   \
    --chunk_size_threshold=0.2