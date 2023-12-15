import numpy as np
import math
import matplotlib.pyplot as plt



class Particle_Filter():

    def __init__(self):
        # Parameters
        self.num_particles = 1000
        self.dt = 0.1  # Time step
        self.motion_noise = 0.01
        self.measurement_noise = 0.1
        self.velocityLimit = 4

    def predict(self,checkPoint,true_pose,particles):
        control = np.array([
                                (checkPoint[0] - true_pose[0])/self.dt,  # Vx
                                (checkPoint[1] - true_pose[1])/self.dt,  # Vy
                                (checkPoint[2] - true_pose[2])/self.dt,  # w
                            ])
        control[0:2] = np.clip(control[0:2],-self.velocityLimit,self.velocityLimit)
        particles = self.motion_model(particles, control)
        return particles
    
    def updateWeights(self,true_pose,particles):
        measurement = np.array([true_pose[0] + np.random.normal(0, self.measurement_noise),
                                    true_pose[1] + np.random.normal(0, self.measurement_noise),
                                    true_pose[2] + np.random.normal(0, self.measurement_noise)])

        # Update particle weights based on measurement model
        weights = self.measurement_model(particles, measurement)
        return weights
    
    def resample(self,weights,particles):
        indices = np.random.choice(np.arange(self.num_particles), self.num_particles, p=weights)
        particles = particles[indices]
        return particles


    def motion_model(self,particles, control):
        particles[:, 0] += control[0] * self.dt  # Update x
        particles[:, 1] += control[1] * self.dt  # Update y
        particles[:, 2] += control[2] * self.dt  # Update w

        # Add motion noise
        particles += np.random.multivariate_normal(np.array([0,0,0]),np.diag([0.001,0.001,0.001]),size=self.num_particles)
        return particles
    
    def calculateError(self, point, line_start, line_end):
        x0, y0 = point[0:2]
        x1, y1 = line_start[0:2]
        x2, y2 = line_end[0:2]

        numerator = (x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)
        denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        distance = numerator / denominator
        return distance

    def measurement_model(self,particles, measurement):
        # Simulated measurement model (replace with your actual sensor model)
        expected_measurement = np.array([particles[:, 0], particles[:, 1], particles[:,2]])

        # Calculate likelihood
        weights = np.exp(-0.5 * np.sum((expected_measurement.T - measurement)**2, axis=1) / self.measurement_noise**2)
        weights /= np.sum(weights)

        return weights

