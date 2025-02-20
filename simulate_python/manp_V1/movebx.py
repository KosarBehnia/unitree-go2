import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

# Load the model

model = mujoco.MjModel.from_xml_path("../../unitree_robots/mnp-v1/box_env.xml")

#model = mujoco.MjModel.from_xml_path("../../unitree_robots/go2/scene.xml")
data = mujoco.MjData(model)

# Default target position
target_position = np.array([0, 0, 0])  # Initial position

# PD Controller Parameters
kp = 50   # Proportional gain
kd = 10   # Derivative gain

# Event to notify about target position update
target_position_updated = threading.Event()

# Function to update target position from user input
def get_user_input():
    global target_position
    while True:
        try:
            # Get user input for the target position
            new_position = input("Enter new target position as 'x y z': ").strip()
            new_position = [float(coord) for coord in new_position.split()]
            
            # Update the target position in the main thread
            target_position[:] = new_position
            target_position_updated.set()  # Notify the simulation to update target position
        except ValueError:
            print("Invalid input. Please enter valid coordinates.")

# Start a thread to handle user input
threading.Thread(target=get_user_input, daemon=True).start()

# Run simulation with the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    while viewer.is_running():
        mujoco.mj_step(model, data)  # Step the simulation

        # Get the current position of the box
        current_position = data.qpos[:3]

        # Compute velocity command using PD control
        position_error = target_position - current_position
        velocity_command = kp * position_error - kd * data.qvel[:3]

        # Apply forces
        data.qvel[:3] = velocity_command * model.opt.timestep  # Smoothly move

        # Update viewer
        viewer.sync()

        # Stop condition
        if np.linalg.norm(position_error) < 0.01:
            break

