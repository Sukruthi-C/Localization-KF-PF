# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
import matplotlib.pyplot as plt
from particle_filter import *
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
    goal_configs = np.array([[-1.2,-1.4,0],
                            [-1.2,1.2,0],
                            [3.4, 1.2,0]])
    
    
    # Waypoints
    draw_sphere_marker((-3.4,-1.4, 1), 0.06, (1, 0, 0, 1))
    draw_sphere_marker((-1.2,-1.4, 1), 0.06, (0, 1, 0, 1))
    draw_sphere_marker((-1.2,1.2, 1), 0.06, (0, 0, 1, 1))
    draw_sphere_marker((3.4, 1.2, 1), 0.06, (0, 0, 1, 1))

    ##############################################################################################################
    ###########################                     PARTICLE FILTER                      ###########################    
    ##############################################################################################################
    
    initial_state = start_config  # Initial state (x, y, heading)
    process_noise = 0.001 # Process noise (velocity, angular velocity, heading change)
    measurement_noise = 0.0001 # Measurement noise covariance (x, y,theta)
    
    # Create Particle filter
    pf = Particle_Filter(process_noise,measurement_noise,initial_state)

    # Store the computed trajectory
    filtered_states = []
    error = []
    resample_threshold = 0.5
    num_steps = 50
    # Compute trajectory for all the waypoints
    prevPoint = start_config
    weights = np.full(pf.num_particles,1.0/pf.num_particles)
    for checkPoint in goal_configs:
        e = 0
        # print("weightsout:",weights)
        # Execute till the robot converges to the target point.
        for step in range(num_steps):
            control_input = pf.get_velocity(checkPoint,prevPoint)
            pf.predict_model(control_input)
            measurement = np.array([prevPoint[0] + np.random.normal(0, pf.measurement_noise),
                                        prevPoint[1] + np.random.normal(0, pf.measurement_noise),
                                        prevPoint[2] + np.random.normal(0,pf.measurement_noise)])
            weights = pf.measurement_model(measurement,weights)
            # print("weights1:",weights)
            # condition check for resampling
            ess = 1/np.sum(np.square(weights))
            # if (ess<resample_threshold):
                # resampling
            idx = pf.residual_resampling(weights)
            new_particles = np.array((pf.particles[idx]))
            pf.particles = new_particles

            # error of pose
            estimated_pose = pf.get_estimated_pose(weights)
            e = np.linalg.norm(prevPoint - estimated_pose)
            filtered_states.append(estimated_pose)
            error.append(e)
            # print("EERRROOORRR:",error)
            if e < 0.01:  
                print("FOUNDDDDD")
                break

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