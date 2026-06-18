from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World
stage = omni.usd.get_context().get_stage()              # 3. Stage

cube_prim_blue = DynamicCuboid(                              # 4. Prim
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=np.array([0.0, 0.0, 1.0]),
    scale=np.array([0.10, 0.10, 0.10]),
    color=np.array([0.0, 0.0, 1.0]),
)

cube_prim_red = DynamicCuboid(                              # 4. Prim
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.0, 0.0, 0.50]),
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([1.0, 0.0, 0.0]),
)





world.scene.add_default_ground_plane()                  # 5. Scene
world.scene.add(cube_prim_blue)
# world.scene.add(cube_prim_red)
# world.scene.add(cube_prim_green)

world.reset()
step_count = 0 

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    step_count += 1

    if step_count % 100 == 0 :
        print(step_count)
    
    if step_count == 300 :
        world.reset()
        world.stop()




    # if step_count == 500 :
    #     simulation_app.close()
    

simulation_app.close()