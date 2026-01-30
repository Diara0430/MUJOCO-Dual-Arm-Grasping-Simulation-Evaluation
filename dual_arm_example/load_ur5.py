import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

xml_path = "assets/dual_arm_and_single_arm/quad_insert.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)  # nu = number of actuators
    ]

    print("Actuator names in order:", actuator_names)

    mujoco.viewer.launch(model, data)
