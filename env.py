# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name,get_joint_info,get_num_joints
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
import pybullet as p
#########################
import particle_filter_test
from helper_fcn import velocity_model,get_omegas
import matplotlib.pyplot as plt
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
    goal_configs = np.array([-2.4,-0.4,np.pi])
    
    trajectory = np.array([[-2.4,-1.4,0],
                            [-2.4,-0.4,np.pi/2],
                            [-3.4,-0.4,np.pi]])
    process_noise = np.array([0.001, 0.001, 0.001])  # Process noise (velocity, angular velocity, heading change)
    measurement_noise = np.array([0.0001, 0.0001, 0.0001])  # Measurement noise covariance (x, y,theta)
    num_steps = 100
    pf = particle_filter_test.Particle_Filter()
    pf.initialize_particles(start_config)
    current_state = start_config
    # print("enteruiung")
    # # get joint indices of wheels
    # num_joints = get_num_joints(robots['pr2'])
    # print("num joints",num_joints)
    # for joint_index in range (num_joints):
    #     # print("num joints",num_joints)
    #     joint_info = get_joint_info(robots['pr2'], joint_index)
    #     # print("joint_info",joint_info)
    #     joint_name = joint_info[1].decode('UTF-8')
    #     # print("       oooooooooooooooo          ")
    #     # print("joint name:",joint_name)
    #     if 'fl_caster_l_wheel_joint' in joint_name:
    #         left_wheel_joint_index = joint_index
    #         print("Left Wheel Joint Index:", left_wheel_joint_index)
    #         l_wheel_info = p.getJointInfo(robots['pr2'],6)
    #         print("left wheel radius:",l_wheel_info[7])

    #     if 'fl_caster_r_wheel_joint' in joint_name:
    #         right_wheel_joint_index = joint_index
    #         print("Right Wheel Joint Index:", right_wheel_joint_index)
    #         r_wheel_info = p.getJointInfo(robots['pr2'],7)
    #         print("left wheel radius:",r_wheel_info[7])

    i=0
    errors = []
    for step in range (num_steps):
        # velocity model is wrong
        omega_l,omega_r = get_omegas(robots['pr2'])
        control = velocity_model(omega_r,omega_l)
        pf.predict_particles(current_state,control,robots['pr2'],measurement_noise)
        pf.low_variance_resample()
        estimate = pf.estimate_state()
        current_state = trajectory[i%len(trajectory)]
        i+=1
        error = pf.calculate_error(estimate,current_state)
        errors.append(error)

    plt.plot(errors)
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.title('Localization Error over Time')
    plt.show()




    # execute_trajectory(robots['pr2'], base_joints, np.array((trajectory)), sleep=0.1)

    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()