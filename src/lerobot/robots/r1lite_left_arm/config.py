from dataclasses import dataclass, field
from typing import Dict

from ..config import RobotConfig


@dataclass
class SimpleCameraSpec:
    width: int
    height: int
    channels: int = 3
    fps: int = 15
    type: str = "passthrough"


@RobotConfig.register_subclass("r1lite_left_arm")
@dataclass
class R1LiteLeftArmRobotConfig(RobotConfig):
    cameras: Dict[str, SimpleCameraSpec] = field(
        default_factory=lambda: {
            "head_left_rgb": SimpleCameraSpec(width=1280, height=720, channels=3),
            "left_wrist_rgb": SimpleCameraSpec(width=1280, height=720, channels=3),
        }
    )

    def feature_shapes(self):
        return {k: (v.height, v.width, v.channels) for k, v in self.cameras.items()}
