import time
from lerobot.robots.kinova_gen3 import KinovaGen3, KinovaGen3Config

cfg = KinovaGen3Config()
cfg.has_gripper = True
print(f"Config has_gripper: {cfg.has_gripper}")
print(f"Config gripper_target_topic: {cfg.gripper_target_topic}")

r = KinovaGen3(cfg)
r.connect()

print(f"\nAfter connect:")
print(f"  pub_gripper_target is None: {r.pub_gripper_target is None}")
print(f"  config.has_gripper: {r.config.has_gripper}")

print("\nSending gripper CLOSE...")
r.send_action({"gripper.pos": 1.0}, wait=False)
time.sleep(1.0)

r.disconnect()
print("Done!")