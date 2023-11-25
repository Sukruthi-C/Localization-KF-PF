


for each timestep:
    move_PR2_along_path(path, motion_noise_params)

    actual_position = get_actual_position_of_PR2()
    simulated_sensor_reading = simulate_sensor_reading(actual_position)

    # Kalman filter estimation
    kalman_estimated_position = kalman_filter(kalman_estimated_state, control_input, simulated_sensor_reading)

    # Particle filter estimation
    particle_estimated_position = particle_filter(particles, control_input, simulated_sensor_reading)

    log_data(actual_position, kalman_estimated_position, particle_estimated_position)

    if check_for_special_conditions():  # Conditions where Kalman fails but Particle filter succeeds
        analyze_special_case()
