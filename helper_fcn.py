# Imports
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Constants
R = np.diag([0.1, 0.1])  # Measurement noise covariance
dt = 0.1 # Seconds
linear_V_limit = 0.1 # m/s
angular_W_limit = 0.01 # rad/s



def motion_planner_model(current_position, target_position):
    # Position: (x,y,theta)
    # Return linear and angular velocity of the robot
    delta_position = (target_position - current_position)
    omega = (math.atan2(delta_position[1], delta_position[0]) + delta_position[2])/dt
    v = np.sqrt(delta_position[0]**2 + delta_position[1]**2)/dt

    return v,omega


 # plotting trajectory and particles
def plot_results(self,true_trajectory, estimated_trajectory, particles, waypoints):
    plt.figure(figsize=(10, 6))
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated Trajectory', color='blue')
    plt.scatter(particles[:, 0], particles[:, 1], s=5, color='red', alpha=0.5, label='Particle Cloud')
    plt.scatter(waypoints[:, 0], waypoints[:, 1], marker='X', s=100, color='black', label='Waypoints')
    plt.title('Particle Filter Localization with Waypoints')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

# plotting trajectories for different speeds
def plot_diff_speed(self,all_errors):
    plt.figure(figsize=(12, 6))

    for speed, errors in all_errors.items():
        plt.plot(errors, label=f"Speed {speed} m/s")

    plt.xlabel("Time Step")
    plt.ylabel("Trajectory Error")
    plt.title("Trajectory Error Over Time for Different Speeds")
    plt.legend()
    plt.grid(True)
    plt.show()

# plotting particle distribution for each speed
def plot_particle_distribution(self, speeds, all_distributions):
    # Creating subplots for each key moment
    fig, axes = plt.subplots(nrows=len(speeds), ncols=1, figsize=(15, 5 * len(speeds)))
    fig.suptitle('Particle Distributions at Different Speeds')
    colors = {'start': 'green', 'mid': 'blue', 'end': 'red'}
    axes = np.array(axes).reshape(-1)

    # Plotting the distributions
    for i, speed in enumerate(speeds):
        for moment in ['start', 'mid', 'end']:
            particles = all_distributions[speed][moment]
            if particles is not None:
                axes[i].scatter(particles[:, 0], particles[:, 1], s=5, alpha=0.5, color=colors[moment], label=f"{moment.capitalize()} particles")
        
        axes[i].set_title(f'Speed {speed} m/s')
        axes[i].set_xlabel('X-axis')
        axes[i].set_ylabel('Y-axis')
        axes[i].legend()
        axes[i].grid(True)

    # Show the plot after creating all subplots
    plt.show()

def plot_cov(mean,cov,plot_axes):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(xy=(mean[0],mean[1]),
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=np.rad2deg(np.arccos(v[0, 0])))
    #ell.set_facecolor('none')
    ell.set_facecolor((1.0, 1.0, 1.0, 0))
    ell.set_edgecolor((0, 0, 0, 1))
    plot_axes.add_artist(ell)
    plt.scatter(mean[0],mean[1],c='r',s=5)

        
# plotting no of steps to reach waypoint
def plot_steps_to_waypoint(self,waypoints,steps_data):
    waypoint_indices = np.arange(len(waypoints))  # Use indices for waypoints
    for speed, steps in steps_data.items():
        plt.plot(waypoint_indices, steps, marker='o', label=f'Speed {speed}')

    plt.xlabel("Waypoint Index")
    plt.ylabel("Steps to Reach Waypoint")
    plt.title("Steps to Reach Waypoints for Different Speeds")
    plt.xticks(waypoint_indices)  # Set x-ticks to waypoint indices
    plt.legend()
    plt.grid(True)
    plt.show()