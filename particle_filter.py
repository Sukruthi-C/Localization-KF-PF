import numpy as np
import numpy as np
from scipy.stats import multivariate_normal
from helper_fcn import predict_next_state,angle_diff

class particle_filter():

    def __init__(self):
        self.num_particles = 1000
        self.weight = np.zeros(1000)

    # here initial uncertainity means how certain are you that 
    # you are at that initial position
    def initialize_particles(self, initial_position):
        # each particle has (x,y,theta)
        # initialize n particles with noise for x,y,theta
        particles = [initial_position + np.random.normal(0, 1, 3) for _ in range(self.num_particles)]
        return particles
    
    # implement kidnapped robot situation
    
    # my control input here is the v,w input. using this and prior predict the posteriori
    def predict_particles(self,particles, control_input):
        v,w = control_input
        # Update each particle's state based on the control input and motion model
        for i, particle in enumerate(particles):
            predicted_state = predict_next_state(particle,dt = 0.02,v=v,omega=w) + np.random.normal(0, 0.6, 3)
            particles[i] = predicted_state
        return particles
    
   
    # update using predicted and sensor reading
    def update_particles(self,particles, measured_position):
        """measured_position"""
        weights = []
        for particle in particles:
            # Compute the probability of the measured position given the particle's position
            prob = self.calculate_measurement_probability(measured_position, particle)
            weights.append(prob)
        # Normalize the weights
        weights /= np.sum(weights)
        return weights

# resampling maintains effective sample size
# effective sample size is the num of particles contribute
# to the posterior distribution

# effective sample size is low then it means that there is a lot of uncertainity in the
# particles and resampling helps this
    def resample_particles(self,particles, weights):
        # Resample particles based on their weights
        new_particles = np.random.choice(particles, size=len(particles), replace=True, p=weights)
        return new_particles
 
    def calculate_measurement_probability(self,measured_position, particle_position):
        
        xs,ys,ts = measured_position #this is the sensor measurement
        xp,yp,tp = particle_position #this is the particle position
        sigma = 0.5 #for linear dimension
        theta_diff = angle_diff(ts,tp)
        prob_x = (1/np.sqrt(2*np.pi*(sigma)**2))*np.exp(-((xs-xp)**2)/(2*(sigma)**2))
        prob_y = (1/np.sqrt(2*np.pi*(sigma)**2))*np.exp(-((ys-yp)**2)/(2*(sigma)**2))
        sigma = 0.7 #for angular dimension
        prob_theta = (1/np.sqrt(2*np.pi*(sigma)**2))*np.exp(-((theta_diff)**2)/(2*(sigma)**2))
        probability = prob_x * prob_y *prob_theta

        return probability

