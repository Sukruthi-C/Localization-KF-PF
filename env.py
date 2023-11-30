# Imports
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
#########################
from helper_fcn import motion_planner,motion_planner_model,getMeasuredPosition
from particle_filter import particle_filter
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
    
    trajectory,actions = motion_planner(start_config,goal_configs)
                    # [-2.4,-0.4,np.pi/2]])
    # print(trajectory)
    # print("robot",robots)

    # TEST for particle filter
    pf = particle_filter()
    particles = pf.initialize_particles(start_config)
    for waypoint in trajectory:
        v,w = motion_planner_model(waypoint,goal_configs)
        action = v,w
        particles = pf.predict_particles(particles,action)
        measured_pos = getMeasuredPosition(robots['pr2'])
        weights = pf.update_particles(particles,measured_pos)
        for particle in particles:
            draw_sphere_marker((particle[0], particle[1], 1), 0.05, (0, 0, 0, 1)) 
        # TODO: Fix Resampling
        # particles = pf.resample_particles(particles,weights)






    
    # kalmanFilter_error = kalman_filter(start_config, goal_configs)
    
    execute_trajectory(robots['pr2'], base_joints, np.array((trajectory)), sleep=0.1)

    # time.sleep(1000)
   
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()