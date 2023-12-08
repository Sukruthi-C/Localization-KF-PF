import numpy as np
from scipy.stats import multivariate_normal
from helper_fcn import predict_next_state,angle_diff,getMeasuredPosition

class particle_filter():

    def __init__(self):
        self.num_particles = 100
        # wrong
        self.weight = np.zeros(1000)

    # here initial uncertainity means how certain are you that 
    # you are at that initial position
    def initialize_particles(self, initial_position):
        # each particle has (x,y,theta)
        # initialize n particles with noise for x,y,theta
        particles = [initial_position + np.random.normal(0, 1, 3) for _ in range(self.num_particles)]
        print("particles initialized")
        return particles
    
    # implement kidnapped robot situation
    # print and check the pybullet position with the actual position i think something is wrong here
    
    # my control input here is the v,w input. using this and prior predict the posteriori
    # TODO:remove robot
    def predict_particles(self,particles, control_input):
        v,w = control_input
    
        # Update each particle's state based on the control input and motion model
        for i, particle in enumerate(particles):
            print("goin to predict next state")
            predicted_state = predict_next_state(particle,dt = 0.02,v=v,omega=w) + np.random.normal(0,[0.2,0.2,0.07])
            particle = predicted_state
        print("predicted the particles")
        return particles
    
   
    # update using predicted and sensor reading
    def update_particles(self,particles, measured_position,measurement_noise):
        """measured_position"""
        weights = []
        for particle in particles:
            # Compute the probability of the measured position given the particle's position
            print("measured position:",measured_position)
            prob = self.calculate_measurement_probability(measured_position, particle,measurement_noise)
            weights.append(prob)
        # Normalize the weights
        weights /= np.sum(weights)
        print("weights calculated")
        # print("weights=",weights)
        return weights

# resampling maintains effective sample size
# effective sample size is the num of particles contribute
# to the posterior distribution

# effective sample size is low then it means that there is a lot of uncertainity in the
# particles and resampling helps this
# SSTART HERE
    def resample_particles(self,particles, weights):
        # residual sampling 
        # Resample particles based on their weights
        N = len(particles)
        indices = list(range(N))
        print("starting resampling")
        # Step 1: Calculate the number of residuals
        num_residuals = np.floor(N * weights).astype(int)
        num_remaining = N - np.sum(num_residuals)
        
        # Step 2: Add residual copies to the new particle set
        new_particles = []
        for idx, count in enumerate(num_residuals):
            new_particles.extend([particles[idx]] * count)
        
        # Step 3: Calculate remaining weights
        remaining_weights = (weights - num_residuals / N) * N / num_remaining
        
        # Step 4: Normalize remaining weights
        remaining_weights /= np.sum(remaining_weights)
        
        # Step 5: Stochastic resampling for the remaining particles
        remaining_indices = np.random.choice(indices, size=num_remaining, p=remaining_weights)
        stochastic_particles = [particles[idx] for idx in remaining_indices]
        
        # Step 6: Combine both sets
        new_particles.extend(stochastic_particles)
        
        print("new particles created")
        # new_particles = np.random.choice(particles, size=len(particles), replace=True, p=weights)
        return new_particles
 
    def calculate_measurement_probability(self,measured_position, particle_position,measurement_noise):
        
        xs,ys,ts = measured_position #this is the sensor measurement
        xp,yp,tp = particle_position #this is the particle position
        sigma_x,sigma_y,sigma_theta = measurement_noise
        theta_diff = angle_diff(ts,tp)
        prob_x = (1/np.sqrt(2*np.pi*(sigma_x)**2))*np.exp(-((xs-xp)**2)/(2*(sigma_x)**2))
        prob_y = (1/np.sqrt(2*np.pi*(sigma_x)**2))*np.exp(-((ys-yp)**2)/(2*(sigma_y)**2))
        prob_theta = (1/np.sqrt(2*np.pi*(sigma_theta)**2))*np.exp(-((theta_diff)**2)/(2*(sigma_theta)**2))
        probability = prob_x * prob_y *prob_theta

        return probability
    
    

