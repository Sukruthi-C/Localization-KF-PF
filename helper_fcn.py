# Imports
import numpy as np
import math

# Constants
R = np.diag([0.1, 0.1])  # Measurement noise covariance
dt = 0.1 # Seconds
linear_V_limit = 0.1 # m/s
angular_W_limit = 0.01 # rad/s

# Public Functions
def simulate_sensor_reading(actual_position):
    # Systematic Error
    systematic_error = calculate_systematic_error()

    # Random Error
    random_error = generate_random_error()

    # Environmental Factors (optional, based on your project's complexity)
    environmental_error = calculate_environmental_error()

    # Combine errors with the actual position
    noisy_position = actual_position + random_error

    return noisy_position

def calculate_systematic_error():
    # This function calculates any constant or predictable errors
    systematic_error_value = None
    # This could be based on calibration data or known sensor biases
    return systematic_error_value

def generate_random_error():
    # Generate random noise, possibly using a Gaussian distribution
    # You can use sensor_noise_params to set the mean and standard deviation

    # Generate noise with 0 mean and R cov
    random_error_value = np.random.multivariate_normal([0, 0], R)
    return random_error_value

def calculate_environmental_error(environmental_factors):
    # If applicable, calculate errors based on environmental factors
    environmental_error_value = None
    # This could include temperature, humidity, electromagnetic interference, etc.
    return environmental_error_value


def motion_planner_model(current_position, target_position):
    # Position: (x,y,theta)
    # Return linear and angular velocity of the robot
    delta_position = (target_position - current_position)/dt
    omega = (math.atan2(delta_position[1], delta_position[0]) + delta_position[2])/dt
    v = np.sqrt(delta_position[0]**2 + delta_position[1]**2)

    return v,omega

def motion_planner(current_position, target_position):

    """ From a start position, it gives the control inputs and positions along the trajectory to the destination.
    To calculate it, it uses RTR method, which stands for Rotate-Translate-Rotate
    R: Rotate from current state to go to the destination position.
    T: Translate to the destination position.
    R: Once reached, rotate on the spot to point to the desination orientation.

    Returns the trajectory of points and the action states
    """

    trajectory = np.empty((0, 3))
    actions = np.empty((0, 2))

    # Rotate on the spot: "R"-T-R
    print("ROTATING")
    current_state = current_position
    angle_to_rotate = math.atan2(target_position[1]-current_position[1],target_position[0]-current_position[0]) - current_position[2]
    end_state = np.array([current_position[0],current_position[1],angle_to_rotate])
    trajectory = np.vstack((trajectory,current_state))
    while (np.linalg.norm(current_state-end_state)> 0.01):
        u = velocity_model(current_state,end_state)
        u[0] = 0
        # True state update
        A = np.eye(3)
        B = np.array([[math.cos(current_state[2]), 0],
                        [math.sin(current_state[2]), 0],
                        [0, 1]])
        next_state = A @ current_state + B @ u
        trajectory = np.vstack((trajectory,next_state))
        actions = np.vstack((actions,u))
        current_state = next_state
    
    
    # Translate towards the destination: R-"T"-R
    print("TRANSLATING")
    end_state = np.array([target_position[0],target_position[1],current_state[2]])
    while (np.linalg.norm(current_state-end_state)> 0.02):
        u = velocity_model(current_state,end_state)
        u[1] = 0

        # True state update
        A = np.eye(3)
        B = np.array([[math.cos(current_state[2]), 0],
                        [math.sin(current_state[2]), 0],
                        [0, 1]])
        next_state = A @ current_state + B @ u
        trajectory = np.vstack((trajectory,next_state))
        actions = np.vstack((actions,u))
        current_state = next_state

    
    # Rotate on the spot: R-T-"R"
    print("ROTATING")
    angle_to_rotate = target_position[2] - current_state[2]
    end_state = target_position
    while (np.linalg.norm(current_state-end_state)> 0.02):
        u = velocity_model(current_state,end_state)
        u[0] = 0

        # True state update
        A = np.eye(3)
        B = np.array([[math.cos(current_state[2]), 0],
                        [math.sin(current_state[2]), 0],
                        [0, 1]])
        next_state = A @ current_state + B @ u
        trajectory = np.vstack((trajectory,next_state))
        actions = np.vstack((actions,u))
        current_state = next_state
    
    
    # print(trajectory)
    # print(actions)
    return trajectory, actions

def velocity_model(current_position,target_position):
    # Calculate the linear and angular velocity of the robot
    v,omega = motion_planner_model(current_position,target_position)
    v_limited = np.clip(v,-linear_V_limit,linear_V_limit)
    omega_limited = np.clip(omega,-angular_W_limit,angular_W_limit)

    # Control Input to the robot
    u = np.array([v_limited, omega_limited])
    return u