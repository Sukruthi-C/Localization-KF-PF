import numpy as np
import pybullet as p


class Particle_Filter():

    def __init__(self):
        self.num_particles = 100
        self.weight = np.full(self.num_particles, 1.0 / self.num_particles)
        self.particles = None

    def initialize_particles(self,initial_pos):
        self.particles = np.array([initial_pos + np.random.normal(0,1,3) for _ in range (self.num_particles)])


    def predict_next_state(self,current_state,dt,v,omega):
        """
        current_state: current position and orientation of the robot 
                        (x,y,theta)
        dt: small change in time/step
        v: linear velocity
        omega: angular velocity"""
        # measured_pos,measured_= getMeasuredPosition(robot)

        x,y,theta = current_state
        d_theta = omega*dt
        theta_new = theta + d_theta
        if abs(omega) < 1e-6:
                omega = 1e-6
                d_x = v*dt*np.cos(theta)
                d_y = v*dt*np.sin(theta)
        else:
            # when the robot is turning
            d_x = (v/omega)*(np.sin(theta_new)-np.sin(theta))
            d_y = (v/omega)*(-np.cos(theta_new)+np.cos(theta))

        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        x_new = d_x + x
        y_new = d_y + y

        next_state = np.array([x_new,y_new,theta_new])

        return next_state
    
    def getMeasuredPosition(self,robot):
        pos,ori = p.getBasePositionAndOrientation(robot)
        noisy_pos = np.array(pos)[:2] + np.random.normal(0, 1, 2)
        noisy_ori = np.array(ori)[2] + np.random.normal(0, 1, 1)
        return np.concatenate((noisy_pos,noisy_ori))
    
    def angle_diff(self,ori1,ori2):
        diff = ori1-ori2

        while (diff > np.pi):
            diff-=2*np.pi
        
        while (diff< -np.pi):
            diff+=2*np.pi

        return diff
    
    def update_particles(self,measured_position, particle_position,measurement_noise):
        xs,ys,ts = measured_position #this is the sensor measurement
        xp,yp,tp = particle_position #this is the particle position
        sigma_x,sigma_y,sigma_theta = measurement_noise
        theta_diff = self.angle_diff(ts,tp)
        prob_x = (1/np.sqrt(2*np.pi*(sigma_x)**2))*np.exp(-((xs-xp)**2)/(2*(sigma_x)**2))
        prob_y = (1/np.sqrt(2*np.pi*(sigma_x)**2))*np.exp(-((ys-yp)**2)/(2*(sigma_y)**2))
        prob_theta = (1/np.sqrt(2*np.pi*(sigma_theta)**2))*np.exp(-((theta_diff)**2)/(2*(sigma_theta)**2))
        probability = prob_x * prob_y *prob_theta

        return probability

    def predict_particles(self,state,control_input,rob,meas_noise):
        v,w = control_input
        particles = np.zeros_like(self.particles)
        weights = np.zeros_like(self.particles)
        dt_ = 0.2
        sum=0
        for i,particle in enumerate (self.particles):
            particles[i] = self.predict_next_state(state,dt_,v,w)
            # get measured pos and meaasured noise
            meas_pos = self.getMeasuredPosition(rob)
            weights[i] = self.update_particles(meas_pos,particles[i],meas_noise)
            sum+=weights[i]
        self.weight = weights/sum
        self.particles = particles

    def low_variance_resample(self):
        new_particles = []
        rand = np.random.uniform(0,1/self.num_particles)
        threshold = (1/self.num_particles)*0.6
        c = self.weight[0]
        i=0
        
        for m in range (self.num_particles):
            u = rand + m*(1/self.num_particles)
            while (u>c).all():
                i+=1
                c+=self.weight[i%self.num_particles]
            new_particles.append(self.particles[i%self.num_particles])
        new_weight = [1/self.num_particles for _ in range (self.num_particles)]
        self.weight = new_weight
        self.particles = new_particles


    def estimate_state(self):
        indices = np.argsort(self.weight)[::-1]
        top_particles = indices[:int(self.num_particles * 0.4)]  # Take top 40% indices

        top_particle_positions = [self.particles[i] for i in top_particles]
        top_particle_weight = [self.weight[i] for i in top_particles]
        estimated_position = np.average(top_particle_positions, axis=0, weights=top_particle_weight)
        return estimated_position
    
    def calculate_error(self,estimated_state, actual_state):
        # Position error
        estimated_position = np.array(estimated_state[:2])  # [x, y] of estimated state
        actual_position = np.array(actual_state[:2])  # [x, y] of actual state
        position_error = np.linalg.norm(estimated_position - actual_position)

        # Orientation error
        estimated_theta = estimated_state[2]
        actual_theta = actual_state[2]
        orientation_error = np.arctan2(np.sin(estimated_theta - actual_theta), np.cos(estimated_theta - actual_theta))

        return position_error, orientation_error
                

                
   

