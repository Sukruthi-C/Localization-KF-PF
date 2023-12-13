import numpy as np
import matplotlib.pyplot as plt


class Particle_Filter():
    def __init__(self):
        # Parameters
        self.num_particles = 1000
        self.dt = 0.1  # Time step
        self.motion_noise = 0.01
        self.measurement_noise = 0.1
        self.desired_speed = 0.7

    def predict_model(self,particles, control):
        particles[:, 0] += control[0] * self.dt  # Update x
        particles[:, 1] += control[1] * self.dt  # Update y

        # Add motion noise
        particles[:, 0] += np.random.normal(0, self.motion_noise, self.num_particles)
        particles[:, 1] += np.random.normal(0, self.motion_noise, self.num_particles)
        return particles


    def measurement_model(self,particles, measurement):
        # Simulated measurement model (replace with your actual sensor model)
        expected_measurement = np.array([particles[:, 0], particles[:, 1]])

        # Calculate likelihood
        weights = np.exp(-0.5 * np.sum((expected_measurement.T - measurement)**2, axis=1) / self.measurement_noise**2)
        weights /= np.sum(weights)

        return weights

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


    def particle_filter_localization(self,num_steps, waypoints):
        # Initialize particles randomly
        particles = np.random.rand(self.num_particles, 2) * np.array([0,0])
        key_module = {'start':None,'mid':None,'end':None}
        true_pose = np.array([0, 0, 0])
        true_trajectory = []
        error = []
        start = 0
        mid = num_steps//2
        end = num_steps-1
        estimated_trajectory = []#np.zeros((num_steps, 3))
        steps = []
        for current_waypoint in waypoints:

            for step in range(num_steps):

                # Calculate control inputs to reach the current waypoint
                dx = current_waypoint[0] - true_pose[0]
                dy = current_waypoint[1] - true_pose[1]
                distance = np.sqrt(dx**2 + dy**2)

                control = np.array([
                    self.desired_speed*dx/distance,  # Vx
                    self.desired_speed*dy/distance,  # Vy
                ])

            
                # Move particles according to motion model
                particles = self.predict_model(particles, control)
                # print("particles from func:",particles)

                # Simulate measurement (replace with your actual sensor measurements)
                measurement = np.array([true_pose[0] + np.random.normal(0, self.measurement_noise),
                                        true_pose[1] + np.random.normal(0, self.measurement_noise)])

                # Update particle weights based on measurement model
                weights = self.measurement_model(particles, measurement)

                # get the particle distribution
                if step == start:
                    key_module['start'] = particles

                elif step == mid:
                    key_module['mid'] = particles

                # Resample particles based on weights
                indices = np.random.choice(np.arange(self.num_particles), self.num_particles, p=weights)
                particles = particles[indices]

                # Estimate the robot's pose based on particle positions (mean or weighted mean)
                estimated_pose = np.mean(particles, axis=0)
                e = np.linalg.norm(true_pose[:2] - estimated_pose[:2])
                error.append(e)

                # Store true and estimated poses
                true_trajectory.append(true_pose[0:2])
                estimated_trajectory.append(estimated_pose)

                # why is true pose = estimated pose?
                true_pose = estimated_pose
                if np.linalg.norm(estimated_pose[0:2] - current_waypoint) < 0.01:  
                    key_module['end'] = particles
                    break
                
            steps.append(step)
                    
                    
            print("\n\n *************************** \n \n")

        error = np.array(error)
        estimated_trajectory = np.array(estimated_trajectory)
        true_trajectory = np.array(true_trajectory)
        waypoints = np.array(waypoints)
        # self.plot_results(true_trajectory, estimated_trajectory, particles, waypoints)
        return error,key_module,steps
    

def main():
    waypoints = [[2, 0],[2, 2],[0, 2]]
    noise_levels = [0.01, 0.05, 0.1]
    # Run the particle filter localization for a specified number of steps
    p = Particle_Filter()
    desired_speeds = [1,3,5]
    all_errors = {}
    all_distribution = {}
    all_steps = {}
    
    for speed in desired_speeds:
        error,distribution,steps = p.particle_filter_localization(num_steps=50, waypoints=waypoints)
        all_errors[speed] = error
        all_distribution[speed] = distribution
        all_steps[speed] = steps

    # p.plot_diff_speed(all_errors)
    # p.plot_particle_distribution(desired_speeds,all_distribution)
    waypoints = np.array(waypoints)
    p.plot_steps_to_waypoint(waypoints,all_steps)


if __name__ == '__main__':
    main()




