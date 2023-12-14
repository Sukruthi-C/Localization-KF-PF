import numpy as np
import matplotlib.pyplot as plt
from particle_filter import Particle_Filter
    

def main():
    waypoints = [[-3.4,-1.4,1],[-1.2,-1.4,1],[-1.2, 1.2,1],[3.4,1.2,1]]
    noise_levels = [0.01, 0.05, 0.1]
    # Run the particle filter localization for a specified number of steps
    p = Particle_Filter()
    desired_speeds = [1,3,5]
    all_errors = {}
    all_distribution = {}
    all_steps = {}
    
    for speed in desired_speeds:
        error,distribution,steps = p.particle_filter_localization(num_steps=50, waypoints=waypoints)
        # all_errors[speed] = error
        # all_distribution[speed] = distribution
        # all_steps[speed] = steps

    # p.plot_diff_speed(all_errors)
    # p.plot_particle_distribution(desired_speeds,all_distribution)
    # waypoints = np.array(waypoints)
    # p.plot_steps_to_waypoint(waypoints,all_steps)


if __name__ == '__main__':
    main()




