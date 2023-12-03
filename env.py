# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from filter import *
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
    # print(start_config)
    goal_configs = np.array([[-2.4,-1.4,0],
                            [-2.4,-0.4,np.pi/2],
                            [-3.4,-0.4,np.pi]])
    initial_state = start_config  # Initial state (x, y, heading)
    initial_covariance = np.diag([1, 1, 1])  # Initial covariance matrix
    process_noise = [0.001, 0.001, 0.001]  # Process noise (velocity, angular velocity, heading change)
    measurement_noise = np.diag([0.0001, 0.0001, 0.0001])  # Measurement noise covariance (x, y,theta)
    # Create Kalman filter
    kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

    # Simulate robot motion and measurements
    filtered_states = []
    inputs = []
    error = []

    for checkPoint in goal_configs:
        # start = start_config # [-3.4 -1.4  0. ]
        # _,actions = motion_planner(start, checkPoint)
        # print(actions)
        while True:
            control_input = velocity_model(kf.state, checkPoint)
            # print("kf.state=", kf.state)
            # print("target=",checkPoint)
            # print("Control Input=",control_input)
            print("Error=",np.linalg.norm(kf.state - checkPoint))
            # print("\n")
            error.append(np.linalg.norm(kf.state - checkPoint))
            if(np.linalg.norm(kf.state - checkPoint) < 0.05):
                break
            # print("*************************** \n")
            # break

            filtered_states.append(kf.state)
            kf.predict(control_input)

            # Simulate noisy measurements (true position with added noise)
            # measurement_noise = np.random.normal(0, 0.1, 3)
            noise = np.random.multivariate_normal([0, 0, 0], measurement_noise)
            # print("Noise=",noise)
            measurement = np.dot(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), kf.state) + noise
            inputs.append(control_input)
            # print(measurement)

            # Update Kalman filter with measurements
            kf.update(measurement)
        # print("*****************************************************************************************\n")
        # start = checkPoint
        # break
    
    # Plot the results
    filtered_states = np.array(filtered_states)
    actual_states = np.vstack((start_config,goal_configs))
    # print(actual_states)
    inputs = np.array(inputs)
    error = np.array(error)
    # for state in true_states:
    #     print(state)        
    # print(trajectory)
    draw_sphere_marker((-3.4,-1.4, 1), 0.06, (1, 0, 0, 1))
    draw_sphere_marker((-2.4,-1.4, 1), 0.06, (0, 1, 0, 1))
    draw_sphere_marker((-2.4,-0.4, 1), 0.06, (0, 0, 1, 1))
    

    plt.figure(1) # 2 rows, 1 column, first plot
    # plt.figure(figsize=(10, 6))
    plt.title("Robot Trajectory")
    plt.plot(filtered_states[:, 0], filtered_states[:, 1], label='Kalman Path', linestyle='--', marker='o')
    plt.plot(actual_states[:,0], actual_states[:,1], label='True Path', linestyle='-', color='blue', marker='o')
    plt.legend()
    
    plt.figure(2)
    plt.title("Error between current position and target")
    plt.plot(error,linestyle='-',color='red')
    # plt.plot(error[:,1],label='Y axis',linestyle='-',color='blue')
    # plt.plot(error[:,2],label='W axis',linestyle='-',color='blue')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    execute_trajectory(robots['pr2'], base_joints, filtered_states, sleep=0.2)
    
    # kalmanFilter_error = kalman_filter(start_config, goal_configs)
    

    # time.sleep(1000)
   
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()