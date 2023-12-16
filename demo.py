# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from kalman_Filter import * 
from particle_filter import *
import matplotlib.pyplot as plt
from helper_fcn import plot_cov
#########################

def main(number):
    robots = None
    obstacles = None
    # print("********************************************************************")
    # print("**                 Select the obstacle environment                **")
    # print("*********************************************************************")
    # number = input("Enter a number between 1-4 (included):")
    # number = int(number)
    robots,obstacles,goal_configs = create_obs_env(number)
    

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    start_config = np.array(get_joint_positions(robots['pr2'], base_joints))
    initial_state = start_config  # Initial state (x, y, heading)
    initial_covariance = np.diag([1, 1, 1])  # Initial covariance matrix
    process_noise = [0.001, 0.001, 0.001]  # Process noise (velocity, angular velocity, heading change)
    measurement_noise = np.diag([0.0001, 0.0001, 0.0001])  # Measurement noise covariance (x, y,theta)
    
    # Create Kalman filter
    kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise,0.1,0.1)


    # Particle filter
    pf = Particle_Filter()
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
    ax = plt.gca()
    plt.ion()
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
            # Given state xt and control input ut, predict state xt+1
            particles = pf.predict(checkPoint,true_pose,particles)

            # update based on measurement model
            weights = pf.updateWeights(true_pose,particles)

            # resampling
            particles = pf.resample(weights,particles)

            # estimated state xt+1 
            estimated_pose = np.mean(particles, axis=0)
            true_trajectory[step] = true_pose

            # for plotting 
            weighted_mean = np.sum(particles.T*weights,axis=1)/np.sum(weights)
            cov_matrix = np.zeros((particles.shape[1],particles.shape[1]))

            if step%3==0:
                for i in range (particles.shape[0]):
                    diff = (particles[i]-weighted_mean).reshape(-1,1)
                    cov_matrix +=weights[i]*(diff@diff.T)
                cov_matrix /=np.sum(weights)
                plot_cov(estimated_pose[:2],cov_matrix,ax)

            pf_states.append(estimated_pose)
            pf_error.append(pf.calculateError(estimated_pose,true_pose,checkPoint))
            true_pose = estimated_pose

            # estimated state is close to next state
            if np.linalg.norm(estimated_pose[0:2] - checkPoint[0:2]) < 0.05:                
                break

        
        prevPoint = checkPoint
    
    # Kalman Filter: Blue
    # Particel Filter: Green
    # Collision points: Red
        
    # Plot Kalman Filter Path
    collision_flag = False
    for pose in kf_states:
        if(collision_fn(pose)):
            draw_sphere_marker((pose[0],pose[1], 0.5), 0.06, (1, 0, 0, 1))
            collision_flag = True
        else:
            draw_sphere_marker((pose[0],pose[1], 0.5), 0.06, (0, 0, 1, 1))
    if collision_flag:
        print("KALMAN FILTER PATH IS IN COLLISION")
    
    # Plot Particle Filter Path
    collision_flag = False
    for pose in pf_states:
        if(collision_fn(pose)):
            draw_sphere_marker((pose[0],pose[1], 0.8), 0.06, (1, 0, 0, 1))
            collision_flag = True
        else:
            draw_sphere_marker((pose[0],pose[1], 0.8), 0.06, (0, 1, 0, 1))
    if collision_flag:
        print("PARTICLE FILTER PATH IS IN COLLISION")
    
    # Plot the results
    kf_states = np.array(kf_states)
    actual_states = np.vstack((start_config,goal_configs))
    kf_error = np.array(kf_error)
    pf_states = np.array(pf_states)
    pf_error = np.array(pf_error)


    execute_trajectory(robots['pr2'], base_joints, pf_states, sleep=0.2)
    plt.figure(1)
    plt.title("Particle Distribution")
    plt.plot(pf_states[:,0],pf_states[:,1],label='Particle Filter Path',linestyle='-',color='blue')
    plt.plot(actual_states[:,0], actual_states[:,1], label='True Path', linestyle=':', color='red')
    plt.plot(kf_states[:, 0], kf_states[:, 1], label='Kalman Filter Path', linestyle='--',color = 'green')
    plt.legend()

    plt.figure(2)
    plt.title("Robot Trajectory")
    plt.plot(kf_states[:, 0], kf_states[:, 1], label='Kalman Filter Path', linestyle='--',color = 'green', marker='x')
    plt.plot(actual_states[:,0], actual_states[:,1], label='True Path', linestyle='-', color='red', marker='o')
    plt.plot(pf_states[:, 0], pf_states[:, 1], label='Particle Filter Path', linestyle='--',color = 'blue', marker='o')
    plt.legend()
    
    plt.figure(3)
    plt.title("Error between current position and target")
    plt.plot(kf_error,linestyle='-',color='red',label='Kalman Filter Error')
    plt.plot(pf_error,linestyle='-',color='blue',label='Particle Filter Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.ioff()
    time.sleep(5)
    print("Please close MATPLOT windows to run next environment. Do not close the Pybullet simulation\n")
    
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

def create_obs_env(number):
    if number == 1:
        # load robot and obstacle resources
        connect(use_gui=True)
        robots, obstacles = load_env('pr2_env1.json')
        goal_configs = np.array([[-1.2,-1.4,0],
                            [-1.2,1.3,0],
                            [3.4, 1.3,0]])
        # Waypoints
        # draw_sphere_marker((-3.4,-1.4, 1), 0.06, (1, 0, 0, 1))
        # draw_sphere_marker((-1.2,-1.4, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-1.2,1.3, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((3.4, 1.3, 1), 0.06, (0, 0, 1, 1))
        return robots,obstacles,goal_configs
    elif number == 2:
        connect(use_gui=True)
        robots, obstacles = load_env('pr2_env2.json')
        goal_configs = np.array([[-1.2,-1.4,0],
                        [-1.2,1.3,0],
                        [-1.2,2.1,0],
                        [-3.4, 2.2,0]])
        # draw_sphere_marker((-3.4,-1.4, 1), 0.06, (1, 0, 0, 1))
        # draw_sphere_marker((-1.2,-1.4, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-1.2,1.3, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-3.4, 2.5, 1), 0.06, (0, 0, 1, 1))
        return robots,obstacles,goal_configs
    elif number == 3:
        connect(use_gui=True)
        robots, obstacles = load_env('pr2_env3.json')
        goal_configs = np.array([[-1.2,-1.4,0],
                        [-1.2,1.3,0],
                        [0.5, 1.3,0],
                        [0.6,-2.0,0],
                        [3.0,-2.0,0],
                        [3.4,1.5,0]])
        # draw_sphere_marker((-3.4,-1.4, 1), 0.06, (1, 0, 0, 1))
        # draw_sphere_marker((-1.2,-1.4, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-1.2,1.3, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((0.8, -2.0, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((3.0,-2.0, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((3.2, 1.5, 1), 0.06, (0, 0, 1, 1))
        return robots,obstacles,goal_configs
    elif number == 4:
        connect(use_gui=True)
        robots, obstacles = load_env('pr2_env4.json')
        goal_configs = np.array([[-2.0,-1.6,0],
                        [-2.0,-0.5,0],
                        [-3.3,-0.3,0],
                        [-3.3,1.8,0],
                        [0.5,1.8,0],
                        [0.5,0.6,0],
                        [-0.5,-0.7,0],
                        [0.3,-1.7,0],
                        [3.4,-1.7,0]])
        # draw_sphere_marker((-3.4,-1.6, 1), 0.06, (1, 0, 0, 1))
        # draw_sphere_marker((-1.5,-1.6, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-1.5,-0.5, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-3.3, 0.0, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-3.3,1.5, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((0.5, 1.5, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((0.5, 0.6, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((-0.5, -0.7, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((0.3, -1.7, 1), 0.06, (0, 0, 1, 1))
        # draw_sphere_marker((3.4, -1.7, 1), 0.06, (0, 0, 1, 1))
        return robots,obstacles,goal_configs
    else:
        print("Not a valid input")
        return


if __name__ == '__main__':
    print("Runtime for each environment: ~30 seconds. \nTotal runtime: ~2-4 mins\n")
    print("INSTRUCTION: When the plots are visible, only close the plots, keep the pybullet simulation open.\n")
    time.sleep(9)
    for i in range(1, 5):
        print("***********************************************************************")
        print("Running Environment: ", i, "\n")
        main(i)
        
        # Show the plots and wait for the user to close the windows
        plt.show(block=True)
        # Close all figures to prepare for the next environment
        plt.close('all')