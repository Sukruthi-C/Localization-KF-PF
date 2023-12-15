# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from kalman_Filter import * 
from particle_filter import *
import matplotlib.pyplot as plt
#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    # print(kalman_filter(0,0,0))
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    start_config = np.array(get_joint_positions(robots['pr2'], base_joints))
    print(start_config)
    goal_configs = np.array([[-1.2,-1.4,0],
                            [-1.2,1.2,0],
                            [3.4, 1.2,0]])
    
    
    # Waypoints
    draw_sphere_marker((-3.4,-1.4, 1), 0.06, (1, 0, 0, 1))
    draw_sphere_marker((-1.2,-1.4, 1), 0.06, (0, 1, 0, 1))
    draw_sphere_marker((-1.2,1.2, 1), 0.06, (0, 0, 1, 1))
    draw_sphere_marker((3.4, 1.2, 1), 0.06, (0, 0, 1, 1))

    
    initial_state = start_config  # Initial state (x, y, heading)
    initial_covariance = np.diag([1, 1, 1])  # Initial covariance matrix
    process_noise = [0.001, 0.001, 0.001]  # Process noise (velocity, angular velocity, heading change)
    measurement_noise = np.diag([0.0001, 0.0001, 0.0001])  # Measurement noise covariance (x, y,theta)
    
    # Create Kalman filter
    kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise,0.1,0.1)


    # Particle filter
    pf = Particle_Filter()
    # particles = np.random.rand(pf.num_particles, 2) * np.array([-3.4,-1.4])
    particles = np.random.multivariate_normal(np.array([0,0,0]),np.diag([0.01,0.01,0.01]),size=pf.num_particles) + np.array([-3.4,-1.4,0])

    # Store the computed trajectory
    kf_states = []
    kf_error = []
    num_steps = 1000
    pf_states = [np.mean(particles, axis=0)]
    true_trajectory = np.zeros((num_steps, 3))
    pf_error = []

    # Compute trajectory for all the waypoints
    prevPoint = start_config
    true_pose = start_config
    for checkPoint in goal_configs:

        ##############################################################################################################
        ###########################                     KALMAN FILTER                      ###########################    
        ##############################################################################################################
        
        # Execute till the robot converges to the target point.
        while True:
            control_input = kf.velocity_model(kf.state, checkPoint)
            kf_error.append(kf.calculateError(kf.state, prevPoint, checkPoint))
            
            if(np.linalg.norm(kf.state - checkPoint) < 0.1):
                break
            
            kf_states.append(kf.state)

            # Kalman Filter Prediction step.
            kf.predict(control_input)

            # Simulate noisy measurements (true position with added noise)
            noise = np.random.multivariate_normal([0, 0, 0], measurement_noise)
            measurement = kf.state + noise
            
            # Update Kalman filter with measurements
            kf.update(measurement)

        ##############################################################################################################
        ###########################                     PARTICLE FILTER                      ######################### 
        ##############################################################################################################

        for step in range(num_steps):
            control = np.array([
                    (checkPoint[0] - true_pose[0])/pf.dt,  # Vx
                    (checkPoint[1] - true_pose[1])/pf.dt,  # Vy
                    (checkPoint[2] - true_pose[2])/pf.dt,  # w
                ])
            control[0:2] = np.clip(control[0:2],-pf.velocityLimit,pf.velocityLimit)
            # control[2] = np.clip(control[2],-pf.velocityLimit,pf.velocityLimit)

            particles = pf.motion_model(particles, control)
            measurement = np.array([true_pose[0] + np.random.normal(0, pf.measurement_noise),
                                    true_pose[1] + np.random.normal(0, pf.measurement_noise),
                                    true_pose[2] + np.random.normal(0, pf.measurement_noise)])

            # Update particle weights based on measurement model
            weights = pf.measurement_model(particles, measurement)
            indices = np.random.choice(np.arange(pf.num_particles), pf.num_particles, p=weights)
            particles = particles[indices]

            # Estimate the robot's pose based on particle positions (mean or weighted mean)
            estimated_pose = np.mean(particles, axis=0)
            # Store true and estimated poses
            true_trajectory[step] = true_pose
            pf_states.append(estimated_pose)
            pf_error.append(pf.calculateError(estimated_pose,true_pose,checkPoint))
            true_pose = estimated_pose
            if np.linalg.norm(estimated_pose[0:2] - checkPoint[0:2]) < 0.05:                
                break

        
        prevPoint = checkPoint
    
    # Plot the results
    kf_states = np.array(kf_states)
    actual_states = np.vstack((start_config,goal_configs))
    kf_error = np.array(kf_error)
    pf_states = np.array(pf_states)
    pf_error = np.array(pf_error)
    

    plt.figure(1)
    plt.title("Robot Trajectory")
    plt.plot(kf_states[:, 0], kf_states[:, 1], label='Kalman Filter Path', linestyle='--',color = 'green', marker='x')
    plt.plot(actual_states[:,0], actual_states[:,1], label='True Path', linestyle='-', color='red', marker='o')
    plt.plot(pf_states[:, 0], pf_states[:, 1], label='Particle Filter Path', linestyle='--',color = 'blue', marker='o')
    plt.legend()
    
    plt.figure(2)
    plt.title("kf_error between current position and target")
    plt.plot(kf_error,linestyle='-',color='red')
    plt.plot(pf_error,linestyle='-',color='blue')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    execute_trajectory(robots['pr2'], base_joints, pf_states, sleep=0.2)
    



    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
