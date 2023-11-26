# Imports
import numpy as np


# Constants
R = np.diag([0.1, 0.1])  # Measurement noise covariance


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
