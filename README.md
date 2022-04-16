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


### Launch Files
- **drone.launch** :  Brings in only the drone and turtleBot3
- **farm.launch** : Brings in drone in a farm
- **racetrack.launch** : Brings in racetrack with 2 tb3 and a drone ( not working )
- **warehouse.launch** : Ware house with Drone

### Launch Ware house Simulation
- Bring this package into your workspace/src/"clone here"
- Source ROS and workspace
- Build workspace
-Install required packages
    - geographic msgs
    - state publishers
    - gazebo-ros

- Launch worlds and drone with the following command
```
roslaunch drone_basic drone.launch
```
```
roslaunch drone_basic warehouse.launch
```
```
roslaunch drone_basic farm.launch
```
```
roslaunch drone_basic racetrack.launch
```
<<<<<<< HEAD
=======

>>>>>>> 14a7a8c94282b315f7ab5f2ca9ceb7a20a375d62
