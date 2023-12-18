# Robot Localization: Kalman Filter and Particle Filter

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
#### Simulation environments for testing:
There are four simulation environments for testing of Kalman Filter and Particle Filter. The robot’s trajectory is simulated in the PyBullet environment, executing movements toward each target position. Both the Kalman Filter and Particle Filter paths are visualized, with collision points marked in red.
<p align="center">
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/f5eab693-c6c4-487b-a277-c666b3785fdd" width="250" height="200" alt="Simulation Environment 1"/>
   <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/f161eb51-59a6-486b-8c13-6fc82a5d875f" width="250" height="200" alt="Simulation Environment 2"/>
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/8102d410-05bd-45b7-956c-5030bb312a1a" width="250" height="200" alt="Simulation Environment 3"/>
  <img src="https://github.com/Sukruthi-C/Localization-KF-PF/assets/123084588/48c0b1ae-5f17-4873-811c-17581f19438b" width="250" height="200" alt="Simulation Environment 4"/>
</p>
<p align="center">
  Simulation Environment 1 | Simulation Environment 2 | Simulation Environment 3 | Simulation Environment 4
</p>









### Project Status
1. **Current Phase:** Completed. 
2. **Completion Date:** 12-15-2023.

### Development Environment Requirements
1. **Python 3.8+**
2. **NumPy 1.1+**
3. **Matplotlib 3.4+**
4. **PyBullet**
Additional dependencies listed in requirements.txt.

### Installation and Usage Guide
1. To get started, clone the repository.
2. Install dependencies: 
    pip install -r requirements.txt.
3. Run the main script: python3 main.py.

### Python official documentation
1. NumPy documentation - https://numpy.org/doc/
2. Matplotlib documentation - https://matplotlib.org/stable/index.html
3. PyBullet Quickstart Guide - https://pythonhosted.org/pybullet/


### Bug Reporting: Please report issues via GitHub issues.
1. **Feature Enhancements:** Suggestions and pull requests are welcome.
2. **Advancing Changes:** Follow the standard GitHub fork-and-pull request workflow.
3. **Known Bugs and Fixes**
Currently, no known bugs. Please report any issues discovered.

### FAQ Section
1. **Q:** What are the key differences between KF and PF in this project?
    A: KF is best for scenarios with Gaussian noise, while PF excels in non-linear, non-Gaussian environments.
2. **Q:** How can I adjust the noise levels in the simulation?
    A: Noise levels can be adjusted in the respective filter classes in the code.

University of Michigan, Ann Arbor, 2023.

