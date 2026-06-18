from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import time
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, Gf

USD_PATH = "/home/doritos/rokey_cobot3/src/basic/taehwan/M0609/collected_m0609_gripper/Collected_m0609_camera/m0609_gripper.usd"

# /World prim 명시적 생성 후 USD reference 연결
stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
world_prim = stage.GetPrimAtPath("/World")
world_prim.GetReferences().AddReference(USD_PATH)


# ---------------------------------------------------------
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")

# 광원 밝기
dome_light.CreateIntensityAttr(1000.0)

# 노출 보정값
dome_light.CreateExposureAttr(0.0)

# 광원 색상: 흰색
dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


for _ in range(15):
    simulation_app.update()

# # 로드된 prim 구조 출력
# print("\n" + "=" * 60)
# print(USD_PATH)
# print("Stage prim 구조")
# print("=" * 60)
# for prim in Usd.PrimRange(stage.GetPseudoRoot()):
#     depth = len(str(prim.GetPath()).split("/")) - 2
#     indent = "  " * depth
#     print(f"{indent}{prim.GetName()}  [{prim.GetTypeName()}]")

# print("\n시뮬레이션 실행 중 (Play 버튼을 눌러 확인하세요)")

while simulation_app.is_running():
    simulation_app.update()
    time.sleep(0.016)

simulation_app.close()
