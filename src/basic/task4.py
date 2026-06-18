from isaacsim import SimulationApp

# ROS 2 Bridge는 다른 Isaac Sim 모듈을 import하기 전에 활성화한다.
simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "RayTracedLighting",
    }
)

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import random
import sys
import time
from typing import Dict, Optional

import numpy as np
import omni.usd
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.sensors.camera import Camera

# task4.py에서는 rclpy를 직접 사용하지 않는다.
# ROS 통신은 Isaac Sim ROS 2 Bridge의 C++ OmniGraph 노드가 담당한다.


_THIS_DIR = Path(__file__).resolve().parent

# rmpflow 인프라 폴더 경로 등록
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from taehwan.M0609.rmpflow.m0609_pick_place_controller import PickPlaceController


# ============================================================
# 1. 로봇 / RMPFlow 설정
# ============================================================
USD_PATH = (
    "/home/doritos/rokey_cobot3/src/basic/taehwan/M0609/"
    "collected_m0609_gripper/Collected_m0609_camera/m0609_gripper.usd"
)
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME = "link_6"
GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]

M0609_URDF_PATH = (
    "/home/doritos/rokey_cobot3/src/basic/taehwan/M0609/"
    "doosan-robot2/urdf/m0609_isaac_sim.urdf"
)
M0609_DESCRIPTION_PATH = (
    "/home/doritos/rokey_cobot3/src/basic/taehwan/M0609/"
    "rmpflow/m0609_description.yaml"
)
M0609_RMPFLOW_CONFIG_PATH = (
    "/home/doritos/rokey_cobot3/src/basic/taehwan/M0609/"
    "rmpflow/m0609_rmpflow_common.yaml"
)

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN = [0.0, 0.0]
GRIPPER_CLOSE = [0.5, 0.5]
GRIPPER_DELTA = [-0.5, -0.5]

FINGER_STATIC = 1.8
FINGER_DYNAMIC = 1.4
CUBE_STATIC = 1.2
CUBE_DYNAMIC = 1.0

EE_OFFSET = np.array([0.0, 0.0, 0.20])

EVENTS_DT = [
    0.008,   # 0. Pick 위치 접근
    0.005,   # 1. 하강
    0.020,   # 2. 그리퍼 닫기 대기
    0.100,   # 3. 그리퍼 닫힘 유지
    0.0025,  # 4. 들어올리기
    0.010,   # 5. Place 위치 이동
    0.0025,  # 6. Place 위치 하강
    1.000,   # 7. 그리퍼 열기 대기
    0.008,   # 8. 상승
    0.080,   # 9. 복귀
]


# ============================================================
# 2. 큐브 / 마커 / 랜덤 스폰 설정
# ============================================================
COLOR_NONE = 0
COLOR_BLUE = 1
COLOR_GREEN = 2

COLOR_NAMES = {
    COLOR_NONE: "NONE",
    COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GREEN",
}

CUBE_SIZE = 0.05
CUBE_Z = CUBE_SIZE / 2.0 + 0.001

PICK_X_RANGE = (-0.1, 0.12)
PICK_Y_RANGE = (0.50, 0.65)

BLUE_PLACE_X_RANGE = (0.45, 0.60)
BLUE_PLACE_Y_RANGE = (0.48, 0.30)

GREEN_PLACE_X_RANGE = (-0.45, -0.60)
GREEN_PLACE_Y_RANGE = (0.48, 0.30)

MIN_CUBE_DISTANCE = 0.14
MIN_GOAL_DISTANCE = 0.16
MAX_RANDOM_SAMPLE_TRIES = 500

GOAL_Z = 0.0
GOAL_MARKER_Z = 0.001

CUBE_COLORS = {
    COLOR_BLUE: np.array([0.0, 0.0, 1.0]),
    COLOR_GREEN: np.array([0.0, 1.0, 0.0]),
}


# ============================================================
# 3. Wrist Camera / ROS 2 토픽 설정
# ============================================================
CAMERA_TOPIC = "wrist_camera/rgb"          # 실제 ROS 토픽: /wrist_camera/rgb
COLOR_RESULT_TOPIC = "/detected_cube_color"
ROS2_COLOR_GRAPH_PATH = "/World/ROS2ColorResultGraph"
ROS2_COLOR_SUBSCRIBER_PATH = (
    f"{ROS2_COLOR_GRAPH_PATH}/ColorSubscriber"
)

CAMERA_FREQUENCY = 15
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAME_ID = "wrist_camera"

FALLBACK_CAMERA_TRANSLATION = np.array([0.0, -0.08, 0.08])
FALLBACK_CAMERA_ROTATION_DEG = np.array([180.0, 0.0, 0.0])


# ============================================================
# 4. 실행 흐름 설정
# ============================================================
HOME_WAIT_FRAMES = 30
CUBE_SETTLE_FRAMES = 20
INTER_CYCLE_WAIT_FRAMES = 45

COLOR_IGNORE_SECONDS = 0.75
COLOR_STABLE_MESSAGES = 3

RANDOM_SEED: Optional[int] = None


# ============================================================
# 유틸리티
# ============================================================
def find_prim_path_by_name(root_path: str, name: str) -> Optional[str]:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def find_camera_prim_path(root_path: str) -> Optional[str]:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    candidates = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Camera):
            path = str(prim.GetPath())
            lower = path.lower()

            score = 0
            if "wrist" in lower:
                score += 100
            if "camera" in lower:
                score += 50
            if "cam" in lower:
                score += 10
            if EE_LINK_NAME.lower() in lower:
                score += 20

            candidates.append((score, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    print("\n  [Camera 후보]")
    for score, path in candidates:
        print(f"    score={score:3d}  {path}")

    return candidates[0][1]


def create_fallback_wrist_camera(ee_path: str) -> str:
    stage = omni.usd.get_context().get_stage()
    camera_path = f"{ee_path}/wrist_camera"

    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    camera_prim.CreateFocalLengthAttr(18.0)
    camera_prim.CreateHorizontalApertureAttr(20.955)
    camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

    xformable = UsdGeom.Xformable(camera_prim.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(
            float(FALLBACK_CAMERA_TRANSLATION[0]),
            float(FALLBACK_CAMERA_TRANSLATION[1]),
            float(FALLBACK_CAMERA_TRANSLATION[2]),
        )
    )
    xformable.AddRotateXYZOp().Set(
        Gf.Vec3f(
            float(FALLBACK_CAMERA_ROTATION_DEG[0]),
            float(FALLBACK_CAMERA_ROTATION_DEG[1]),
            float(FALLBACK_CAMERA_ROTATION_DEG[2]),
        )
    )

    print("\n  [경고] USD 내부에서 Camera 프림을 찾지 못했습니다.")
    print(f"  [생성] 보조 Wrist Camera = {camera_path}")
    return camera_path


def initialize_robot(robot: SingleManipulator, world: World) -> None:
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    robot.set_joint_positions(np.zeros(robot.num_dof))


def set_cube_pose_and_zero_velocity(
    cube: DynamicCuboid,
    position: np.ndarray,
) -> None:
    cube.set_world_pose(
        position=np.asarray(position, dtype=float),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))


def sample_separated_positions(
    rng: random.Random,
    x_range,
    y_range,
    z: float,
    count: int,
    minimum_distance: float,
):
    positions = []
    for _ in range(count):
        for _attempt in range(MAX_RANDOM_SAMPLE_TRIES):
            candidate = np.array(
                [
                    rng.uniform(*x_range),
                    rng.uniform(*y_range),
                    z,
                ],
                dtype=float,
            )

            if all(
                np.linalg.norm(candidate[:2] - previous[:2]) >= minimum_distance
                for previous in positions
            ):
                positions.append(candidate)
                break
        else:
            raise RuntimeError(
                "무작위 위치 생성 실패: 영역을 넓히거나 최소 간격을 줄이세요."
            )
    return positions


def publish_rgb(camera: Camera, frequency: int):
    render_product = camera._render_product_path
    step_size = max(1, int(round(60 / frequency)))

    render_var = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.Rgb.name
    )
    writer = rep.writers.get(render_var + "ROS2PublishImage")
    writer.initialize(
        frameId=CAMERA_FRAME_ID,
        nodeNamespace="",
        queueSize=1,
        topicName=CAMERA_TOPIC,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        render_var + "IsaacSimulationGate",
        render_product,
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    print(f"  [ROS2 PUB] /{CAMERA_TOPIC} @ 약 {frequency} Hz")
    return writer


# ============================================================
# ROS 2 색상 결과 구독: rclpy 대신 Isaac Sim OmniGraph 사용
# ============================================================
def create_color_result_subscriber_graph():
    keys = og.Controller.Keys

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(ROS2_COLOR_GRAPH_PATH).IsValid():
        stage.RemovePrim(ROS2_COLOR_GRAPH_PATH)
        simulation_app.update()

    og.Controller.edit(
        {
            "graph_path": ROS2_COLOR_GRAPH_PATH,
            "evaluator_name": "execution",
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ColorSubscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
            ],
            keys.SET_VALUES: [
                ("ColorSubscriber.inputs:messagePackage", "std_msgs"),
                ("ColorSubscriber.inputs:messageSubfolder", "msg"),
                ("ColorSubscriber.inputs:messageName", "Int32"),
                ("ColorSubscriber.inputs:topicName", COLOR_RESULT_TOPIC),
                ("ColorSubscriber.inputs:queueSize", 10),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ColorSubscriber.inputs:execIn"),
            ],
        },
    )

    simulation_app.update()
    simulation_app.update()

    data_attribute_path = f"{ROS2_COLOR_SUBSCRIBER_PATH}.outputs:data"
    data_attribute = og.Controller.attribute(data_attribute_path)

    try:
        _ = data_attribute.get()
    except Exception as exc:
        raise RuntimeError(
            "ROS2Subscriber의 outputs:data를 만들지 못했습니다. "
            "isaacsim.ros2.bridge 확장이 정상 활성화됐는지 확인하세요."
        ) from exc

    print(f"  [ROS2 SUB/OmniGraph] {COLOR_RESULT_TOPIC} (std_msgs/msg/Int32)")
    return data_attribute


class ColorResultMonitor:
    def __init__(self, data_attribute):
        self._data_attribute = data_attribute
        self._ignore_until = 0.0
        self._last_value = COLOR_NONE
        self._same_count = 0
        self._stable_value = COLOR_NONE

    def reset_for_new_cube(self) -> None:
        self._ignore_until = time.monotonic() + COLOR_IGNORE_SECONDS
        self._last_value = COLOR_NONE
        self._same_count = 0
        self._stable_value = COLOR_NONE

    def update(self) -> None:
        try:
            raw_value = self._data_attribute.get()
            value = int(raw_value)
        except (TypeError, ValueError):
            return

        if value not in (COLOR_NONE, COLOR_BLUE, COLOR_GREEN):
            return

        if time.monotonic() < self._ignore_until:
            return

        if value == COLOR_NONE:
            self._last_value = COLOR_NONE
            self._same_count = 0
            self._stable_value = COLOR_NONE
            return

        if value == self._last_value:
            self._same_count += 1
        else:
            self._last_value = value
            self._same_count = 1

        if self._same_count >= COLOR_STABLE_MESSAGES:
            self._stable_value = value

    @property
    def stable_value(self) -> int:
        return self._stable_value


# ============================================================
# Task
# ============================================================
class M0609ColorTask(BaseTask):
    def __init__(self, name: str, random_seed: Optional[int] = None):
        super().__init__(name=name, offset=None)
        self._robot: Optional[SingleManipulator] = None
        self._camera_path: Optional[str] = None
        self._cubes: Dict[int, DynamicCuboid] = {}
        self._goal_markers: Dict[int, VisualCuboid] = {}

        self._layout_rng = random.Random(random_seed)

        cube_positions = sample_separated_positions(
            rng=self._layout_rng,
            x_range=PICK_X_RANGE,
            y_range=PICK_Y_RANGE,
            z=CUBE_Z,
            count=2,
            minimum_distance=MIN_CUBE_DISTANCE,
        )
        
        
        
        blue_goal_position = np.array(
            [
                self._layout_rng.uniform(*BLUE_PLACE_X_RANGE),
                self._layout_rng.uniform(*BLUE_PLACE_Y_RANGE),
                GOAL_Z,
            ],
            dtype=float,
        )

        green_goal_position = np.array(
            [
                self._layout_rng.uniform(*GREEN_PLACE_X_RANGE),
                self._layout_rng.uniform(*GREEN_PLACE_Y_RANGE),
                GOAL_Z,
            ],
            dtype=float,
        )

        self._goal_positions = {
            COLOR_BLUE: blue_goal_position,
            COLOR_GREEN: green_goal_position,
        }

        self._cube_start_positions = {
            COLOR_BLUE: cube_positions[0],
            COLOR_GREEN: cube_positions[1],
        }

        self._goal_marker_positions = {
            color_id: np.array(
                [
                    goal_position[0],
                    goal_position[1],
                    GOAL_MARKER_Z,
                ],
                dtype=float,
            )
            for color_id, goal_position in self._goal_positions.items()
        }

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links_and_camera()
        self._setup_physics()
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n  [완료] Task 4 씬 구성 성공\n")

    def _load_usd(self) -> None:
        print("\n" + "=" * 70)
        print("[1.LOAD] M0609 USD 로드")
        print("=" * 70)

        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()

        world_prim.GetReferences().AddReference(USD_PATH)

        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {USD_PATH}")

    def _discover_links_and_camera(self) -> None:
        print("\n" + "=" * 70)
        print("[2.DISCOVER] 링크 및 Wrist Camera 탐색")
        print("=" * 70)

        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}'을(를) 찾지 못했습니다.")

        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

        for joint_name in GRIPPER_JOINTS:
            joint_path = find_prim_path_by_name(ROBOT_PRIM_PATH, joint_name)
            print(f"  {joint_name:<35} = {joint_path}")

        self._camera_path = find_camera_prim_path(ROBOT_PRIM_PATH)
        if self._camera_path is None:
            self._camera_path = create_fallback_wrist_camera(self._ee_path)

        print(f"  Wrist Camera = {self._camera_path}")

    def _setup_physics(self) -> None:
        print("\n" + "=" * 70)
        print("[3.PHYSICS] 로봇 Drive 설정")
        print("=" * 70)

        stage = omni.usd.get_context().get_stage()
        drive_count = 0

        root_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        for prim in Usd.PrimRange(root_prim):
            for drive_type in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1

        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene) -> None:
        print("\n" + "=" * 70)
        print("[4.REGISTER] 로봇 등록")
        print("=" * 70)

        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )

        print(f"  [OK] SingleManipulator = {ROBOT_PRIM_PATH}")

    def _create_scene(self, scene) -> None:
        print("\n" + "=" * 70)
        print("[5.SCENE] 파란/초록 큐브와 마커 생성")
        print("=" * 70)

        cube_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=CUBE_STATIC,
            dynamic_friction=CUBE_DYNAMIC,
            restitution=0.0,
        )

        self._cubes[COLOR_BLUE] = scene.add(
            DynamicCuboid(
                prim_path="/World/blue_cube",
                name="blue_cube",
                position=self._cube_start_positions[COLOR_BLUE],
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=CUBE_COLORS[COLOR_BLUE],
                mass=0.05,
                physics_material=cube_material,
            )
        )

        self._cubes[COLOR_GREEN] = scene.add(
            DynamicCuboid(
                prim_path="/World/green_cube",
                name="green_cube",
                position=self._cube_start_positions[COLOR_GREEN],
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=CUBE_COLORS[COLOR_GREEN],
                mass=0.05,
                physics_material=cube_material,
            )
        )

        self._goal_markers[COLOR_BLUE] = scene.add(
            VisualCuboid(
                prim_path="/World/blue_goal_marker",
                name="blue_goal_marker",
                position=self._goal_marker_positions[COLOR_BLUE],
                scale=np.array([0.07, 0.07, 0.002]),
                color=CUBE_COLORS[COLOR_BLUE],
            )
        )

        self._goal_markers[COLOR_GREEN] = scene.add(
            VisualCuboid(
                prim_path="/World/green_goal_marker",
                name="green_goal_marker",
                position=self._goal_marker_positions[COLOR_GREEN],
                scale=np.array([0.07, 0.07, 0.002]),
                color=CUBE_COLORS[COLOR_GREEN],
            )
        )

        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )

        for link_name in ("left_inner_finger", "right_inner_finger"):
            link_path = find_prim_path_by_name(ROBOT_PRIM_PATH, link_name)
            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"{link_name}_geom",
                ).apply_physics_material(finger_material)

    def randomize_and_reset_layout(self) -> None:
        """큐브와 목표 위치를 새로운 무작위 좌표로 갱신하고 즉시 이동시킵니다."""
        print("\n  [RESET] 새로운 무작위 위치를 생성합니다...")
        
        cube_positions = sample_separated_positions(
            rng=self._layout_rng,
            x_range=PICK_X_RANGE,
            y_range=PICK_Y_RANGE,
            z=CUBE_Z,
            count=2,
            minimum_distance=MIN_CUBE_DISTANCE,
        )
        goal_positions = sample_separated_positions(
            rng=self._layout_rng,
            x_range=PLACE_X_RANGE,
            y_range=PLACE_Y_RANGE,
            z=GOAL_Z,
            count=2,
            minimum_distance=MIN_GOAL_DISTANCE,
        )

        self._cube_start_positions[COLOR_BLUE] = cube_positions[0]
        self._cube_start_positions[COLOR_GREEN] = cube_positions[1]
        self._goal_positions[COLOR_BLUE] = goal_positions[0]
        self._goal_positions[COLOR_GREEN] = goal_positions[1]

        self._goal_marker_positions = {
            color_id: np.array(
                [goal_pos[0], goal_pos[1], GOAL_MARKER_Z], dtype=float
            )
            for color_id, goal_pos in self._goal_positions.items()
        }

        for color_id in (COLOR_BLUE, COLOR_GREEN):
            set_cube_pose_and_zero_velocity(
                self._cubes[color_id], self._cube_start_positions[color_id]
            )
            self._goal_markers[color_id].set_world_pose(
                position=self._goal_marker_positions[color_id]
            )

        self.print_random_layout()

    def get_observations(self):
        observations = {
            "m0609_robot": {
                "joint_positions": self.robot.get_joint_positions(),
            }
        }
        for color_id, cube in self._cubes.items():
            position, _ = cube.get_world_pose()
            observations[cube.name] = {
                "position": position,
                "color_id": color_id,
            }
        return observations

    def post_reset(self) -> None:
        if self._robot is not None:
            self._robot.gripper.set_joint_positions(
                self._robot.gripper.joint_opened_positions
            )

    @property
    def robot(self) -> SingleManipulator:
        if self._robot is None:
            raise RuntimeError("Robot이 아직 생성되지 않았습니다.")
        return self._robot

    @property
    def camera_path(self) -> str:
        if self._camera_path is None:
            raise RuntimeError("Camera가 아직 생성되지 않았습니다.")
        return self._camera_path

    def get_cube(self, color_id: int) -> DynamicCuboid:
        return self._cubes[color_id]

    def get_goal_position(self, color_id: int) -> np.ndarray:
        return self._goal_positions[color_id].copy()

    def get_cube_start_position(self, color_id: int) -> np.ndarray:
        return self._cube_start_positions[color_id].copy()

    def print_random_layout(self) -> None:
        print("\n  [현재 무작위 배치]")
        for color_id in (COLOR_BLUE, COLOR_GREEN):
            print(
                f"    {COLOR_NAMES[color_id]:5s} cube = "
                f"{np.round(self._cube_start_positions[color_id], 4)}"
            )
            print(
                f"    {COLOR_NAMES[color_id]:5s} goal = "
                f"{np.round(self._goal_positions[color_id], 4)}"
            )


# ============================================================
# Workflow 보조 함수
# ============================================================
def select_existing_cube(
    task: M0609ColorTask,
    color_id: int,
) -> DynamicCuboid:
    cube = task.get_cube(color_id)
    cube_position, _ = cube.get_world_pose()

    print("\n" + "-" * 70)
    print(f"[SELECT] 기존 큐브 = {COLOR_NAMES[color_id]}")
    print(f"[SELECT] 현재 Pick 위치 = {np.round(cube_position, 4)}")
    print("[SELECT] 순간이동 없이 현재 위치에서 Pick을 시작합니다.")
    print("-" * 70)

    return cube


def verify_placement(
    task: M0609ColorTask,
    actual_cube_color: int,
    detected_color: int,
) -> None:
    cube = task.get_cube(actual_cube_color)
    cube_position, _ = cube.get_world_pose()
    target = task.get_goal_position(detected_color)

    xy_error = float(np.linalg.norm(cube_position[:2] - target[:2]))
    z_error = float(abs(cube_position[2] - CUBE_Z))

    print("\n" + "=" * 70)
    print("[CYCLE 완료]")
    print(f"  실제 큐브 색상 = {COLOR_NAMES[actual_cube_color]}")
    print(f"  영상 판별 색상 = {COLOR_NAMES[detected_color]}")
    print(f"  선택된 목표     = {target}")
    print(f"  최종 큐브 위치  = {cube_position}")
    print(f"  XY 오차         = {xy_error:.4f} m")
    print(f"  Z 오차          = {z_error:.4f} m")

    if actual_cube_color != detected_color:
        print("  [경고] 영상 색상 오분류로 다른 마커가 선택되었습니다.")
    elif xy_error <= 0.08:
        print("  [성공] 같은 색상 마커에 배치되었습니다.")
    else:
        print("  [경고] 색상은 맞지만 배치 위치 오차가 큽니다.")
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main() -> None:
    my_world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )

    task = M0609ColorTask(
        name="m0609_color_task",
        random_seed=RANDOM_SEED,
    )
    my_world.add_task(task)
    my_world.reset()

    robot = task.robot
    initialize_robot(robot, my_world)

    print("\n" + "=" * 70)
    print("[CAMERA] Wrist Camera 초기화 및 ROS 2 이미지 발행")
    print("=" * 70)

    wrist_camera = Camera(
        prim_path=task.camera_path,
        name="wrist_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=CAMERA_RESOLUTION,
    )
    wrist_camera.initialize()
    simulation_app.update()
    wrist_camera.initialize()

    rgb_writer = publish_rgb(wrist_camera, CAMERA_FREQUENCY)
    _ = rgb_writer

    color_data_attribute = create_color_result_subscriber_graph()
    color_monitor = ColorResultMonitor(color_data_attribute)

    print("\n" + "=" * 70)
    print("[CONTROLLER] PickPlaceController 생성")
    print("=" * 70)

    controller = PickPlaceController(
        name="m0609_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )

    print("  [OK] Controller 생성 완료")
    print("\n[준비] Isaac Sim의 Play 버튼을 누르세요.")
    print("[필수] PC B에서 m0609_color_detector.py를 먼저 실행하세요.\n")

    rng = random.Random(RANDOM_SEED)

    was_playing = False
    workflow_state = "idle"
    wait_frames = 0

    color_order = [COLOR_BLUE, COLOR_GREEN]
    cycle_index = 0

    actual_cube_color = COLOR_NONE
    active_cube: Optional[DynamicCuboid] = None
    locked_detected_color = COLOR_NONE

    last_event = None
    last_wait_message_time = 0.0

    try:
        while simulation_app.is_running():
            my_world.step(render=True)
            color_monitor.update()

            is_playing = my_world.is_playing()

            if is_playing and not was_playing:
                print("\n" + "#" * 70)
                print("[RUN] Task 4 Workflow 시작")
                print("#" * 70)

                my_world.reset()
                initialize_robot(robot, my_world)
                wrist_camera.initialize()
                controller.reset()

                task.print_random_layout()

                color_order = [COLOR_BLUE, COLOR_GREEN]
                rng.shuffle(color_order)

                cycle_index = 0
                actual_cube_color = COLOR_NONE
                active_cube = None
                locked_detected_color = COLOR_NONE

                wait_frames = HOME_WAIT_FRAMES + CUBE_SETTLE_FRAMES
                workflow_state = "home_wait"
                last_event = None

                print(
                    "  처리 순서 = "
                    + " -> ".join(COLOR_NAMES[color_id] for color_id in color_order)
                )

            if is_playing and workflow_state == "home_wait":
                wait_frames -= 1
                if wait_frames <= 0:
                    workflow_state = "prepare_cube"

            elif is_playing and workflow_state == "prepare_cube":
                if cycle_index >= len(color_order):
                    workflow_state = "finished"
                else:
                    actual_cube_color = color_order[cycle_index]
                    active_cube = select_existing_cube(
                        task=task,
                        color_id=actual_cube_color,
                    )

                    controller.reset()
                    color_monitor.reset_for_new_cube()
                    locked_detected_color = COLOR_NONE
                    last_event = None
                    last_wait_message_time = 0.0

                    workflow_state = "pick_place"
                    print("[VISION] Wrist Camera 색상 판별 대기 중")

            elif is_playing and workflow_state == "pick_place":
                if active_cube is None:
                    raise RuntimeError("active_cube가 지정되지 않았습니다.")

                current_event = int(controller.get_current_event())

                if (
                    locked_detected_color == COLOR_NONE
                    and 1 <= current_event <= 4
                    and color_monitor.stable_value in (COLOR_BLUE, COLOR_GREEN)
                ):
                    locked_detected_color = color_monitor.stable_value
                    print(
                        f"\n[VISION LOCK] {COLOR_NAMES[locked_detected_color]} "
                        f"({locked_detected_color})"
                    )
                    print(
                        f"[PLACE TARGET] "
                        f"{task.get_goal_position(locked_detected_color)}\n"
                    )

                cube_position, _ = active_cube.get_world_pose()
                current_joints = robot.get_joint_positions()

                if (
                    locked_detected_color == COLOR_NONE
                    and current_event >= 4
                ):
                    now = time.monotonic()
                    if now - last_wait_message_time >= 2.0:
                        print(
                            "[WAIT] 색상 결과가 없어 Pick 후 대기 중입니다. "
                            f"토픽 확인: {COLOR_RESULT_TOPIC}"
                        )
                        last_wait_message_time = now
                else:
                    if locked_detected_color == COLOR_NONE:
                        placing_position = (
                            task.get_goal_position(COLOR_BLUE)
                            + task.get_goal_position(COLOR_GREEN)
                        ) / 2.0
                    else:
                        placing_position = task.get_goal_position(
                            locked_detected_color
                        )

                    actions = controller.forward(
                        picking_position=cube_position,
                        placing_position=placing_position,
                        current_joint_positions=current_joints,
                        end_effector_offset=EE_OFFSET,
                    )
                    robot.apply_action(actions)

                new_event = int(controller.get_current_event())
                if new_event != last_event:
                    ee_position, _ = robot.end_effector.get_world_pose()
                    print(
                        f"[EVENT {new_event}] "
                        f"cube={np.round(cube_position, 4)} "
                        f"ee={np.round(ee_position, 4)} "
                        f"vision={COLOR_NAMES[locked_detected_color]}"
                    )
                    last_event = new_event

                if controller.is_done():
                    if locked_detected_color == COLOR_NONE:
                        raise RuntimeError(
                            "Controller가 끝났지만 색상 결과가 잠기지 않았습니다."
                        )

                    verify_placement(
                        task=task,
                        actual_cube_color=actual_cube_color,
                        detected_color=locked_detected_color,
                    )

                    cycle_index += 1
                    wait_frames = INTER_CYCLE_WAIT_FRAMES
                    workflow_state = "inter_cycle_wait"

            elif is_playing and workflow_state == "inter_cycle_wait":
                wait_frames -= 1
                if wait_frames <= 0:
                    workflow_state = "prepare_cube"

            elif is_playing and workflow_state == "finished":
                print("\n" + "#" * 70)
                print("[사이클 완료] 큐브와 목표를 리셋하고 새로운 작업을 시작합니다.")
                print("#" * 70 + "\n")
                
                # 1. 큐브와 마커 위치 새로운 랜덤 좌표로 재배치
                task.randomize_and_reset_layout()
                
                # 2. 로봇 홈 위치 복귀 및 컨트롤러/그리퍼 리셋
                robot.set_joint_positions(np.zeros(robot.num_dof))
                robot.gripper.set_joint_positions(robot.gripper.joint_opened_positions)
                controller.reset()
                
                # 3. 작업 순서 재섞기 및 사이클 초기화
                color_order = [COLOR_BLUE, COLOR_GREEN]
                rng.shuffle(color_order)
                cycle_index = 0
                locked_detected_color = COLOR_NONE
                
                print("  새로운 처리 순서 = " + " -> ".join(COLOR_NAMES[cid] for cid in color_order))
                
                # 4. 상태를 초기화하여 대기 후 다시 Pick & Place 시작
                wait_frames = HOME_WAIT_FRAMES + CUBE_SETTLE_FRAMES
                workflow_state = "home_wait"

            was_playing = is_playing

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()