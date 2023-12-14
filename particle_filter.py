import numpy as np
import matplotlib.pyplot as plt


class Particle_Filter():
    def __init__(self,process_noise,measurement_noise,initial_state):
        # Parameters
        self.num_particles = 100
        self.dt = 0.1  # Time step
        self.motion_noise = process_noise
        self.measurement_noise = measurement_noise
        self.desired_speed = 0.7
        self.particles = np.zeros((self.num_particles, 3))  # Add a third column for theta
        
        # Initialize particles
        self.particles[:, 0] = initial_state[0] + np.random.normal(0,process_noise,self.num_particles) # X
        self.particles[:, 1] = initial_state[1] + np.random.normal(0,process_noise,self.num_particles)   # Y
        self.particles[:, 2] = initial_state[2] + np.random.normal(0,process_noise,self.num_particles)   # Theta
        # print("particles:",self.particles)
        


    def predict_model(self, control):
        v = control[0]
        omega = control[1]

        # Add noise to control inputs
        v_noisy = v + np.random.normal(0, self.motion_noise, self.num_particles)
        omega_noisy = omega + np.random.normal(0, self.motion_noise, self.num_particles)

        # Update orientation
        self.particles[:, 2] += omega_noisy * self.dt

        # Update position
        self.particles[:, 0] += v_noisy * np.cos(self.particles[:, 2]) * self.dt
        self.particles[:, 1] += v_noisy * np.sin(self.particles[:, 2]) * self.dt

        # Optional: Limit the angle to [-pi, pi] range
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Debugging print statement
        print("particles:", self.particles)




    def measurement_model(self, measurement,weights):
        # Simulated measurement model (replace with your actual sensor model)
        expected_measurement = np.array([self.particles[:, 0], self.particles[:, 1], self.particles[:, 2]])
        squared_diff = np.sum((expected_measurement.T - measurement)**2, axis=1)

        # Calculate log-likelihoods
        log_weights = -0.5 * squared_diff / (self.measurement_noise**2 + 1e-10)
        max_log_weight = np.max(log_weights)
        log_weights -= max_log_weight
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        return weights

    # plotting trajectory and self.particles
    # def plot_results(self,true_trajectory, estimated_trajectory,particle_list, waypoints):
    #     plt.figure(figsize=(10, 6))
    #     # print("estimated traj:",estim
        
    #     # plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], label='Estimated Trajectory', color='blue')
    #     plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated Trajectory', color='blue')
    #     plt.scatter(particle_list[:, 0], particle_list[:, 1], s=5, color='red', alpha=0.5, label='Particle Cloud')
    #     plt.scatter(waypoints[:, 0], waypoints[:, 1], marker='X', s=100, color='black', label='Waypoints')
    #     plt.title('Particle Filter Localization with Waypoints')
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Y-axis')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # plotting trajectories for different speeds
    # def plot_diff_speed(self,all_errors):
    #     plt.figure(figsize=(12, 6))

    #     for speed, errors in all_errors.items():
    #         plt.plot(errors, label=f"Speed {speed} m/s")

    #     plt.xlabel("Time Step")
    #     plt.ylabel("Trajectory Error")
    #     plt.title("Trajectory Error Over Time for Different Speeds")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # # plotting particle distribution for each speed
    # def plot_particle_distribution(self, speeds, all_distributions):
    #     # Creating subplots for each key moment
    #     fig, axes = plt.subplots(nrows=len(speeds), ncols=1, figsize=(15, 5 * len(speeds)))
    #     fig.suptitle('Particle Distributions at Different Speeds')
    #     colors = {'start': 'green', 'mid': 'blue', 'end': 'red'}
    #     axes = np.array(axes).reshape(-1)

    #     # Plotting the distributions
    #     for i, speed in enumerate(speeds):
    #         for moment in ['start', 'mid', 'end']:
    #             self.particles = all_distributions[speed][moment]
    #             if self.particles is not None:
    #                 axes[i].scatter(self.particles[:, 0], self.particles[:, 1], s=5, alpha=0.5, color=colors[moment], label=f"{moment.capitalize()} self.particles")
            
    #         axes[i].set_title(f'Speed {speed} m/s')
    #         axes[i].set_xlabel('X-axis')
    #         axes[i].set_ylabel('Y-axis')
    #         axes[i].legend()
    #         axes[i].grid(True)

    #     # Show the plot after creating all subplots
    #     plt.show()

            
    # # plotting no of steps to reach waypoint
    # def plot_steps_to_waypoint(self,waypoints,steps_data):
    #     waypoint_indices = np.arange(len(waypoints))  # Use indices for waypoints
    #     for speed, steps in steps_data.items():
    #         plt.plot(waypoint_indices, steps, marker='o', label=f'Speed {speed}')

    #     plt.xlabel("Waypoint Index")
    #     plt.ylabel("Steps to Reach Waypoint")
    #     plt.title("Steps to Reach Waypoints for Different Speeds")
    #     plt.xticks(waypoint_indices)  # Set x-ticks to waypoint indices
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def residual_resampling(self,weights):
        N = len(weights)
        # # Step 1: Normalize weights
        # weights /= np.sum(weights)
        # print("weights:",weights)
        new_indices = np.zeros(N, dtype=np.int32)
        num_children = np.floor(N * weights).astype(int)  # Integer part
        residual = weights - num_children / N  # Residual part
        # print("residual:",residual)
        residual /= np.sum(residual)
        new_indices = np.repeat(np.arange(N), num_children)

        # multinomial resampling for the residual part
        num_residual = N - np.sum(num_children) 
        if num_residual > 0:
            residual_indices = np.random.choice(N, num_residual, p=residual)
            new_indices = np.concatenate([new_indices, residual_indices])

        return new_indices
    
    def get_estimated_pose(self, weights):
        # Sort the particles based on weights in descending order
        sort_idx = np.argsort(weights)[::-1]
        sort_particles = self.particles[sort_idx]
        num_top_particles = int(0.4 * self.num_particles)
        top_particles = sort_particles[:num_top_particles]

        # weighted mean of these top particles
        top_weights = weights[sort_idx][:num_top_particles]
        estimated_pose = np.average(top_particles, axis=0, weights=top_weights)

        return estimated_pose
    
    def get_velocity(self,checkpoint,prevPoint):
        dx = abs(checkpoint[0] - prevPoint[0])
        dy = abs(checkpoint[1] - prevPoint[1])

        # calculate the direction
        distance = np.sqrt(dx**2 + dy**2)
        target_direction = np.arctan2(dy,dx)
        angle_to_target = target_direction - checkpoint[2]
        angle_to_target = (angle_to_target + np.pi) % (2*np.pi) - np.pi
        omega = angle_to_target/self.dt
        v = min(self.desired_speed, distance / self.dt) + np.random.normal(0, self.motion_noise)
        control = np.array([
            v,  # Vx
            omega
        ])

        print("v",v)
        print("w",omega)
        
        return control


    # def particle_filter_localization(self,num_steps, waypoints):
    #     # Initialize self.particles randomly
    #     # self.particles = np.random.rand(self.num_particles, 3) * np.array([0,0])
    #     key_module = {'start':None,'mid':None,'end':None}
    #     pred_pose = np.array([0, 0, 0])
    #     true_trajectory = []
    #     error = []
    #     start = 0
    #     mid = num_steps//2
    #     end = num_steps-1
    #     estimated_trajectory = []#np.zeros((num_steps, 3))
    #     steps = []
    #     particle_list = []
    #     for current_waypoint in waypoints:

    #         for step in range(num_steps):

    #             # # Calculate control inputs to reach the current waypoint
    #             # dx = current_waypoint[0] - pred_pose[0]
    #             # dy = current_waypoint[1] - pred_pose[1]
    #             # distance = np.sqrt(dx**2 + dy**2)
    #             # target_direction = np.arctan2(dy,dx)
    #             # angle_to_target = target_direction - current_waypoint[2]
    #             # angle_to_target = (angle_to_target + np.pi) % (2*np.pi) - np.pi
    #             # omega = angle_to_target/self.dt
    #             # v = min(self.desired_speed, distance / self.dt) + np.random.normal(0, 0.1)
    #             # print("v:",v)
    #             # print("ome",omega)
    #             # control = np.array([
    #             #     v,  # Vx
    #             #     omega
    #             # ])

            
    #             # Move self.particles according to motion model
    #             self.predict_model(control)
    #             # print("self.particles from func:",self.particles)
    #             particle_list.append(self.particles)
    #             # Simulate measurement (replace with your actual sensor measurements)
    #             measurement = np.array([pred_pose[0] + np.random.normal(0, self.measurement_noise),
    #                                     pred_pose[1] + np.random.normal(0, self.measurement_noise),
    #                                     pred_pose[2] + np.random.normal(0,self.measurement_noise)])

    #             # Update particle weights based on measurement model
    #             weights = self.measurement_model(measurement)

    #             # get the particle distribution
    #             if step == start:
    #                 key_module['start'] = self.particles

    #             elif step == mid:
    #                 key_module['mid'] = self.particles

    #             # Resample self.particles based on weights
    #             # indices = np.random.choice(np.arange(self.num_self.particles), self.num_self.particles, p=weights)
    #             indices = self.residual_resampling(weights)
    #             self.particles = np.array(self.particles[indices])
    #             print(len(indices))
    #             # particle_list.append(self.particles)
    #             # Estimate the robot's pose based on particle positions (mean or weighted mean)
    #             # PROB HERE
    #             estimated_pose = self.get_estimated_pose(weights)
    #             e = np.linalg.norm(pred_pose - estimated_pose)
    #             error.append(e)

    #             # Store true and estimated poses
    #             true_trajectory.append(pred_pose[0:2])
    #             estimated_trajectory.append(estimated_pose)

    #             # why is true pose = estimated pose?
    #             # we are estimating the true pose here in localization
    #             pred_pose = estimated_pose
    #             if np.linalg.norm(estimated_pose - current_waypoint) < 0.01:  
    #                 key_module['end'] = self.particles
    #                 break
                
    #         steps.append(step)
                    
                    
    #         print("\n\n *************************** \n \n")

    #     error = np.array(error)
    #     estimated_trajectory = np.array(estimated_trajectory)
    #     true_trajectory = np.array(true_trajectory)
    #     waypoints = np.array(waypoints)
    #     particle_list = np.array(particle_list)
    #     self.plot_results(true_trajectory, estimated_trajectory, particle_list, waypoints)
    #     return error,key_module,step