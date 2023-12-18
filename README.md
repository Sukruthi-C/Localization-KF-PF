# Robot Localization: Kalman Filter and Particle Filter

<p align="center">
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/455f24b1-365e-4048-8739-a3c4b70bc4d3" width="250" height="200" alt="des"/>
   <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/d9e13d52-5e89-43aa-be91-ffeded3c4e65" width="250" height="200" alt="des"/>
</p>



## About:
The quest for accurate localization of autonomous robots forms the bedrock of efficient navigation and decision-making. Kalman and particle filters are widely used in robotics for tasks like robot navigation and mapping. These filters help estimate the robot's position and orientation based on sensor data. This project focuses on the implementation and comparative analysis of Particle and Kalman filters for navigation in a simulated environment. Utilizing a PR2 robot model, the study delves into the challenges of autonomous navigation amidst obstacles, emphasizing the accuracy of trajectory estimation under varying conditions. Autonomous navigation in environments laden with obstacles poses significant challenges in robotics, especially in terms of accurate localization and path planning. To address these challenges, this project implements advanced filtering techniques - Particle filters and Kalman filters - integrated into a PR2 robot navigating within a simulated environment crafted in PyBullet.

## Intructions:
1. Clone the repository. 

```
git clone https://github.com/Sukruthi-C/Localization-KF-PF.git
```
2. Install the pre-requisities.
```
cd Localization-KF-PF
./install.sh
```
3. Run demo.py.
```
python3 demo.py
```

### Results:
The contrasting performance of the Kalman Filter and Par-
ticle Filter underscores the importance of selecting an appro-
priate filtering technique based on the system’s characteristics
and environmental conditions. The Particle Filter’s resilience
in non-linear and unpredictable environments makes it a more
versatile choice for state estimation in complex scenarios.
This simulation study demonstrates that while the Kalman
Filter is effective in linear systems with Gaussian noise, its
limitations become evident in more complex environments.
The Particle Filter, with its flexibility and adaptability, offers
a robust alternative for state estimation in scenarios where the
Kalman Filter falters
#### Simulation environments for testing:
There are four simulation environments for testing of Kalman Filter and Particle Filter. The robot’s trajectory is simulated in the PyBullet environment, executing movements toward each target position. Both the Kalman Filter and Particle Filter paths are visualized in green and blue dots respectively with collision points marked in red.
<p align="center">
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/f5eab693-c6c4-487b-a277-c666b3785fdd" width="230" height="180" alt="Simulation Environment 1"/>
   <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/f161eb51-59a6-486b-8c13-6fc82a5d875f" width="230" height="180" alt="Simulation Environment 2"/>
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/8102d410-05bd-45b7-956c-5030bb312a1a" width="230" height="180" alt="Simulation Environment 3"/>
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/48c0b1ae-5f17-4873-811c-17581f19438b" width="230" height="180" alt="Simulation Environment 4"/>
</p>
<p align="center">
  Simulation Environment 1 | Simulation Environment 2 | Simulation Environment 3 | Simulation Environment 4
</p>


#### Trajectory path for KF and PF:
In Figure 1 both the Kalman and Particle Filter were able
to localize the robot and follow the trajectory. In Figure 2, the
Kalman Filter failed as the robot was too close to the wall,
and the estimated position collided with the wall. These states
are represented with a red dot. In both Figure 2 and Figure 2
the covariance distribution of the particles is shown in black
ellipse. 
<p align="center">
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/b90c0276-8876-4ab9-b1be-15a74598f4ea" width="250" height="200" alt="des"/>
   <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/31d4e6cf-3b62-4a3e-8cb1-06d6cf909d7a" width="250" height="200" alt="des"/>
</p>
<p align="center">
  Figure 1 | Figure 2 
</p>

#### Trajectory error for KF and PF:
Figure 1 and 2 show the Trajectory error of the Kalman
Filter and Particle Filter in both environments. Particle Filter
takes less time to reach the goal than Kalman Filter and the
noise associated with it is also lower
<p align="center">
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/72dd9939-c03c-4864-bcf3-a68aec496695" width="250" height="200" alt="des"/>
   <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/c0b197f4-b055-438e-b96b-e76170109451" width="250" height="200" alt="des"/>
</p>
<p align="center">
  Figure 1 | Figure 2 
</p>




### Youtube:
Link: https://youtube.com/playlist?list=PLp10txzP4kbHfUR9XASin_6qsvqL1lneQ&feature=shared




### Development Environment Requirements
1. **Python 3.8+**
2. **NumPy 1.1+**
3. **Matplotlib 3.4+**
4. **PyBullet**
Additional dependencies listed in requirements.txt.

### Python official documentation
1. NumPy documentation - https://numpy.org/doc/
2. Matplotlib documentation - https://matplotlib.org/stable/index.html
3. PyBullet Quickstart Guide - https://pythonhosted.org/pybullet/


### Bug Reporting: Please report issues via GitHub issues.
1. **Feature Enhancements:** Suggestions and pull requests are welcome.
2. **Advancing Changes:** Follow the standard GitHub fork-and-pull request workflow.
3. **Known Bugs and Fixes**
Currently, no known bugs. Please report any issues discovered.

### Project Status
1. **Current Phase:** Completed. 
2. **Completion Date:** 12-15-2023.

### FAQ Section
1. **Q:** What are the key differences between KF and PF in this project?
    A: KF is best for scenarios with Gaussian noise, while PF excels in non-linear, non-Gaussian environments.
2. **Q:** How can I adjust the noise levels in the simulation?
    A: Noise levels can be adjusted in the respective filter classes in the code.

University of Michigan, Ann Arbor, 2023.

