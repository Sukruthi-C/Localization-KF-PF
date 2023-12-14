# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from kalman_Filter import *
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

    ##############################################################################################################
    ###########################                     KALMAN FILTER                      ###########################    
    ##############################################################################################################
    
    initial_state = start_config  # Initial state (x, y, heading)
    initial_covariance = np.diag([1, 1, 1])  # Initial covariance matrix
    process_noise = [0.001, 0.001, 0.001]  # Process noise (velocity, angular velocity, heading change)
    measurement_noise = np.diag([0.0001, 0.0001, 0.0001])  # Measurement noise covariance (x, y,theta)
    
    # Create Kalman filter
    kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise,0.01,0.1)

    # Store the computed trajectory
    filtered_states = []
    error = []

    # Compute trajectory for all the waypoints
    prevPoint = start_config
    for checkPoint in goal_configs:
        
        # Execute till the robot converges to the target point.
        while True:
            control_input = kf.velocity_model(kf.state, checkPoint)
            error.append(kf.calculateError(kf.state, prevPoint, checkPoint))
            
            if(np.linalg.norm(kf.state - checkPoint) < 0.05):
                break
            
            filtered_states.append(kf.state)

            # Kalman Filter Prediction step.
            kf.predict(control_input)

            # Simulate noisy measurements (true position with added noise)
            noise = np.random.multivariate_normal([0, 0, 0], measurement_noise)
            measurement = kf.state + noise
            
            # Update Kalman filter with measurements
            kf.update(measurement)
        
        prevPoint = checkPoint
    
    # Plot the results
    filtered_states = np.array(filtered_states)
    actual_states = np.vstack((start_config,goal_configs))
    error = np.array(error)
    

    plt.figure(1)
    plt.title("Robot Trajectory")
    plt.plot(filtered_states[:, 0], filtered_states[:, 1], label='Kalman Path', linestyle='--', marker='o')
    plt.plot(actual_states[:,0], actual_states[:,1], label='True Path', linestyle='-', color='blue', marker='o')
    plt.legend()
    
    plt.figure(2)
    plt.title("Error between current position and target")
    plt.plot(error,linestyle='-',color='red')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    execute_trajectory(robots['pr2'], base_joints, filtered_states, sleep=0.2)
    



    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
