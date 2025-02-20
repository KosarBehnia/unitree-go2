import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

# Load the Go2 robot model instead of the box model
model = mujoco.MjModel.from_xml_path("../temp/scene.xml")
data = mujoco.MjData(model)

# Default target position (you can adjust this depending on the robot's design)
target_position = np.array([0, 0, 0.6])  # Initial position for the target

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

        # Get the current position of the robot (usually a link or the base frame)
        current_position = data.qpos[:3]


        # Compute velocity command using PD control
        position_error = target_position - current_position
        velocity_command = kp * position_error - kd * data.qvel[:3]

        # Apply forces (or modify the actuator commands if needed)
        data.qvel[:3] = velocity_command * model.opt.timestep  # Smoothly move

        # Update viewer
        viewer.sync()

        # Stop condition
        if np.linalg.norm(position_error) < 0.01:
            break

