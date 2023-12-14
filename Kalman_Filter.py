# Imports
from helper_fcn import *

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input):
        # Prediction step (Motion model)
        delta_t = 0.1  # Time step
        vx, vy, omega = control_input
        theta = self.state[2]
        
        if abs(omega) < 1e-6:
            omega = 1e-6

        # State transition matrix
        A = np.eye(3)

        # Control input matrix
        B = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

        # Process noise covariance matrix
        Q = np.diag([self.process_noise[0], self.process_noise[1], self.process_noise[2]])

        # Prediction step
        self.state = np.dot(A, self.state) + np.dot(B, control_input)
        self.covariance = np.dot(np.dot(A, self.covariance), A.T) + Q

    def update(self, measurement):
        # Update step (Measurement model)
        H = np.eye(3)

        # Measurement noise covariance matrix
        R = np.diag([self.measurement_noise[0], self.measurement_noise[1], self.measurement_noise[2]])

        # Kalman gain
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(np.dot(np.dot(H, self.covariance), H.T) + R))

        # Update step
        self.state = self.state + np.dot(K, (measurement - np.dot(H, self.state)))
        self.covariance = np.dot((np.eye(3) - np.dot(K, H)), self.covariance)


def particle_filter(particles, control_input, measurement):
    # Implement the steps of the particle filter
    updated_particles = None
    # Use particle_filter_params for particles and noise modeling
    return updated_particles
