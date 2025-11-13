# src/lerobot/robots/r1lite/config_r1lite.py
from dataclasses import dataclass, field
from typing import Dict
from ..config import RobotConfig

@dataclass
class SimpleCameraSpec:
    width: int
    height: int
    channels: int = 3
    fps: int = 15
    type: str = "passthrough"  # 只是占位；你走 ROS 订阅，不实际用

@RobotConfig.register_subclass("r1lite")
@dataclass
class R1LiteRobotConfig(RobotConfig):
    # 明确声明 cameras 字段的类型为“带属性的相机规格”
    cameras: Dict[str, SimpleCameraSpec] = field(
        default_factory=lambda: {
            # 键名务必和 R1LiteRobot.observation_features 使用的一致！
            "head_left_rgb":   SimpleCameraSpec(width=1280, height=720, channels=3),
            "head_right_rgb":  SimpleCameraSpec(width=1280, height=720, channels=3),
            "left_wrist_rgb":  SimpleCameraSpec(width=1280, height=720, channels=3),
            "right_wrist_rgb": SimpleCameraSpec(width=1280, height=720, channels=3),
        }
    )

    def feature_shapes(self):
        # 用属性访问，父类 __post_init__ 也能拿到 .width/.height
        return {k: (v.height, v.width, v.channels) for k, v in self.cameras.items()}

