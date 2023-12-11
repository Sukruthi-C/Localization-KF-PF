import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles = 1000
dt = 0.1  # Time step
motion_noise = 0.01
measurement_noise = 0.1

# Function to update particle positions based on control inputs
def motion_model(particles, control):
    particles[:, 0] += control[0] * dt  # Update x
    particles[:, 1] += control[1] * dt  # Update y
    # particles[:, 2] += control[2] * dt  # Update orientation

    # Add motion noise
    particles[:, 0] += np.random.normal(0, motion_noise, num_particles)
    particles[:, 1] += np.random.normal(0, motion_noise, num_particles)
    # particles[:, 2] += np.random.normal(0, motion_noise, num_particles)

    return particles

# Function to update particle weights based on measurements
def measurement_model(particles, measurement):
    # Simulated measurement model (replace with your actual sensor model)
    expected_measurement = np.array([particles[:, 0], particles[:, 1]])

    # Calculate likelihood
    weights = np.exp(-0.5 * np.sum((expected_measurement.T - measurement)**2, axis=1) / measurement_noise**2)
    # print(expected_measurement)
    # print(np.sum((expected_measurement.T - measurement)**2, axis=1) / measurement_noise**2)
    # Normalize weights
    weights /= np.sum(weights)

    return weights

# Plotting function
def plot_results(true_trajectory, estimated_trajectory, particles, waypoints):
    plt.figure(figsize=(10, 6))

    # Plot true trajectory
    # plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], label='True Trajectory', color='green')

    # Plot estimated trajectory
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated Trajectory', color='blue')

    # Plot particle cloud
    plt.scatter(particles[:, 0], particles[:, 1], s=5, color='red', alpha=0.5, label='Particle Cloud')

    # Plot waypoints
    waypoints = np.array(waypoints)
    plt.scatter(waypoints[:, 0], waypoints[:, 1], marker='X', s=100, color='black', label='Waypoints')

    plt.title('Particle Filter Localization with Waypoints')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main localization loop
def particle_filter_localization(num_steps, waypoints):
    # Initialize particles randomly
    particles = np.random.rand(num_particles, 2) * np.array([0,0])

    # True initial pose (for comparison)
    true_pose = np.array([0, 0, 0])

    # Arrays to store true and estimated trajectories
    true_trajectory = np.zeros((num_steps, 2))
    estimated_trajectory = []#np.zeros((num_steps, 3))
    for current_waypoint in waypoints:
        # print(current_waypoint)
        print("Going to waypoint",current_waypoint)
        for step in range(num_steps):
            # Update waypoints based on your logic
            # current_waypoint = waypoints[step % len(waypoints)]

            # Calculate control inputs to reach the current waypoint
            control = np.array([
                current_waypoint[0] - true_pose[0],  # Vx
                current_waypoint[1] - true_pose[1],  # Vy
            ])
            # print(f"{control=}   --  ")
            # Normalize orientation difference to the range [-pi, pi]
            # control[2] = np.arctan2(np.sin(control[2]), np.cos(control[2]))
            print(f"{control=}")
        
            # Move particles according to motion model
            particles = motion_model(particles, control)

            # Simulate measurement (replace with your actual sensor measurements)
            measurement = np.array([true_pose[0] + np.random.normal(0, measurement_noise),
                                    true_pose[1] + np.random.normal(0, measurement_noise)])

            # Update particle weights based on measurement model
            weights = measurement_model(particles, measurement)
            # print("Particles and weights\n")
            # print(f"{particles=}\n{np.sort(weights)=}")
            # for part in (particles,weights):
            #     print(part)

            # Resample particles based on weights
            # print(weights)
            indices = np.random.choice(np.arange(num_particles), num_particles, p=weights)
            particles = particles[indices]

            # Estimate the robot's pose based on particle positions (mean or weighted mean)
            estimated_pose = np.mean(particles, axis=0)

            # Store true and estimated poses
            true_trajectory[step] = true_pose[0:2]
            # estimated_trajectory[step] = estimated_pose
            estimated_trajectory.append(estimated_pose)

            # Print or visualize the results
            # print(f"Step {step + 1}: True Pose: {true_pose}, Estimated Pose: {estimated_pose}\n")
            true_pose = estimated_pose
            # true_pose[0] += control[0] * np.cos(true_pose[2]) * dt  # Update x
            # true_pose[1] += control[1] * np.sin(particles[:, 2]) * dt  # Update y
            # true_pose[:, 2] += control[2] * dt  # Update orientation
            # print(f"{estimated_pose=}\n{current_waypoint=}")
            print(f"{estimated_pose=}")
            print(np.linalg.norm(estimated_pose[0:2] - current_waypoint))
            if np.linalg.norm(estimated_pose[0:2] - current_waypoint) < 0.01:                
                break
        print("\n\n *************************** \n \n")
    
    estimated_trajectory = np.array(estimated_trajectory)
    # print(estimated_trajectory)
            

    # Plot the results
    plot_results(true_trajectory, estimated_trajectory, particles, waypoints)

# Define waypoints for the robot to follow
waypoints = [
    [2, 0],
    [2, 2],
    [0, 2],
    # [2, 2]
]

# Run the particle filter localization for a specified number of steps
particle_filter_localization(num_steps=50, waypoints=waypoints)
