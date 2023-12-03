import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input):
        # Prediction step (Motion model)
        delta_t = 1.0  # Time step
        v, omega = control_input
        theta = self.state[2]
        
        if abs(omega) < 1e-6:
            omega = 1e-6

        # State transition matrix
        F = np.array([[1, 0, -v / omega * np.cos(theta) + v / omega * np.cos(theta + omega * delta_t)],
                      [0, 1, -v / omega * np.sin(theta) + v / omega * np.sin(theta + omega * delta_t)],
                      [0, 0, 1]])

        # Control input matrix
        B = np.array([[1 / omega * (-np.sin(theta) + np.sin(theta + omega * delta_t)),
                       v / omega * (-1 / omega * np.cos(theta) + 1 / omega * np.cos(theta + omega * delta_t))],
                      [1 / omega * (np.cos(theta) - np.cos(theta + omega * delta_t)),
                       v / omega * (-1 / omega * np.sin(theta) + 1 / omega * np.sin(theta + omega * delta_t))],
                      [0, delta_t]])

        # Process noise covariance matrix
        Q = np.diag([self.process_noise[0], self.process_noise[1], self.process_noise[2]])

        # Prediction step
        self.state = np.dot(F, self.state) + np.dot(B, control_input)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, measurement):
        # Update step (Measurement model)
        H = np.array([[1, 0, 0], [0, 1, 0]])

        # Measurement noise covariance matrix
        R = np.diag([self.measurement_noise[0], self.measurement_noise[1]])

        # Kalman gain
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(np.dot(np.dot(H, self.covariance), H.T) + R))

        # Update step
        self.state = self.state + np.dot(K, (measurement - np.dot(H, self.state)))
        self.covariance = np.dot((np.eye(3) - np.dot(K, H)), self.covariance)

# Simulation parameters
num_steps = 100
initial_state = np.array([0, 0, 0])  # Initial state (x, y, heading)
initial_covariance = np.diag([1, 1, 1])  # Initial covariance matrix
process_noise = [0.01, 0.01, 0.01]  # Process noise (velocity, angular velocity, heading change)
measurement_noise = [0.1, 0.1]  # Measurement noise (x, y)

# Create Kalman filter
kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

# Simulate robot motion and measurements
true_states = []
measurements = []
for _ in range(num_steps):
    # Simulate robot motion (constant velocity and heading change)
    control_input = [0.1, 0.0]  # [velocity, angular velocity]
    true_states.append(kf.state)
    kf.predict(control_input)

    # Simulate noisy measurements (true position with added noise)
    measurement_noise = np.random.normal(0, 0.1, 2)
    measurement = np.dot(np.array([[1, 0, 0], [0, 1, 0]]), kf.state) + measurement_noise
    measurements.append(measurement)

    # Update Kalman filter with measurements
    kf.update(measurement)

# Plot the results
true_states = np.array(true_states)
measurements = np.array(measurements)

plt.figure(figsize=(10, 6))
plt.plot(true_states[:, 0], true_states[:, 1], label='True Path', linestyle='--', marker='o')
plt.plot(measurements[:, 0], measurements[:, 1], label='Measurements', linestyle='', marker='x')
plt.plot(kf.state[0], kf.state[1], label='Estimated Path', linestyle='-', marker='s')
plt.title('Kalman Filter Localization for Differential Drive Robot')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
