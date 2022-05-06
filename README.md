# ROS2 Based Drone Ai and Computer vision Course ( Under Development )


- This repository is bieng build on ROS1 ( future version will shift to ROS2 )

### Teaching Process
- **Basic Starting**
    - Setup ROS and vscode
    - Obtain hector Drone Package
    - Understanding Drone Package launch files
    - Setup drone into your package
    - Create nodes for drone
        - Fly drone service
        - Camera recording
        - Go to goal
    - Launch files for empty world , examplery world with cubes
- **Ware House and Farm House**
    - Start by creating a basic model
    - Giving image ( random ) and bringing it into gazebo by creating sdf
    - concept of adding a texture from poly heaven
    - Adding Light then gazebo repeat process to make it feel easy
    - Parts specific images like boxes etc
    - solidfy and array concepts

- **Race Track**
    - Model obtaining from repository
    - World and models integerating in own package
    - Multi Robot Spawning -> groups
    - ROS bags for driving robots

### Installation
sudo apt-get install ros-noetic-turtlebot3-gazebo
sudo apt-get install ros-noetic-turtlebot3-simulations
sudo apt-get install ros-noetic-teleop-twist-keyboard
sudo apt-get install ros-foxy-joint-state-publisher
sudo apt-get install ros-foxy-robot-state-publisher
sudo apt-get install ros-foxy-gazebo-plugins


### Launch Files
- **drone.launch** :  Brings in only the drone and turtleBot3
- **farm.launch** : Brings in drone in a farm
- **racetrack.launch** : Brings in racetrack with 2 tb3 and a drone ( not working )
- **warehouse.launch** : Ware house with Drone

### Launch Ware house Simulation
- Bring this package into your workspace/src/"clone here"
- Source ROS and workspace
- Build workspace
- Install required packages
    - geographic msgs
    - state publishers
    - gazebo-ros


- Launch worlds and drone with the following command
```
roslaunch intelligent_drone drone.launch
```
```
roslaunch intelligent_drone warehouse.launch
```
```
roslaunch intelligent_drone farm.launch
```
```
roslaunch intelligent_drone racetrack.launch
```