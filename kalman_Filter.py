import numpy as np
import math

# Constants
R = np.diag([0.01, 0.01, 0.01])  # Measurement noise covariance
dt = 0.1 # Seconds

class KalmanFilter:
    def __init__(self, 
                 initial_state, 
                 initial_covariance, 
                 process_noise, 
                 measurement_noise,
                 linear_V_limit,
                 angular_W_limit):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.linear_V_limit = linear_V_limit
        self.angular_W_limit = angular_W_limit

    def simulate_sensor_reading(self, actual_position):
        # Systematic Error
        # systematic_error = calculate_systematic_error()

        # Random Error
        random_error = np.random.multivariate_normal([0, 0, 0], R)
        # Environmental Factors (optional, based on your project's complexity)
        # environmental_error = calculate_environmental_error()

        # Combine errors with the actual position
        noisy_position = actual_position + random_error

        return noisy_position

    def motion_planner_model(self, current_position, target_position):
        # Position: (x,y,theta)
        # Return linear and angular velocity of the robot
        delta = target_position - current_position
        delta = delta/dt

        return delta

    def velocity_model(self, current_position,target_position):
        # Calculate the linear and angular velocity of the robot
        delta = self.motion_planner_model(current_position,target_position)
        theta = current_position[2]
        trans = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        
        delta = np.linalg.inv(trans) @ delta
        # print("Delta=",delta)
        delta[0] = np.clip(delta[0],-self.linear_V_limit,self.linear_V_limit)
        delta[1] = np.clip(delta[1],-self.linear_V_limit,self.linear_V_limit)
        delta[2] = np.clip(delta[2],-self.angular_W_limit,self.angular_W_limit)

        return delta
    
    def calculateError(self, point, line_start, line_end):
        x0, y0 = point[0:2]
        x1, y1 = line_start[0:2]
        x2, y2 = line_end[0:2]

        numerator = (x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)
        denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        distance = numerator / denominator
        return distance

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

