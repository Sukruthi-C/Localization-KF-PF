import numpy as np


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