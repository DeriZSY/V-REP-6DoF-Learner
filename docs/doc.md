# V-REP

## Available built-in robot arm models (not complete)

* Dobot Magician
* FrankaEmikaPanda
* KUKA LBR iiwa 14 R820
* KUKA LBR iiwa 7 R800
* KUKA LBR4+
* UR10
* UR3
* UR5

> For more available models, open V-REP and explore the model browser

## Import model from CAD data

V-REP (or CoppeliaSim) supports currently following CAD data formats: [OBJ](https://www.coppeliarobotics.com/helpFiles/en/importExport.htm), [STL](https://www.coppeliarobotics.com/helpFiles/en/importExport.htm), [DXF](https://www.coppeliarobotics.com/helpFiles/en/importExport.htm), [3DS](https://www.coppeliarobotics.com/helpFiles/en/importExport.htm) (Windows only), [Collada](https://www.coppeliarobotics.com/helpFiles/en/colladaPlugin.htm) and [URDF](https://www.coppeliarobotics.com/helpFiles/en/urdfPlugin.htm)

> More detailed information can be found in the official [tutorial](https://www.coppeliarobotics.com/helpFiles/en/buildingAModelTutorial.htm)

## Quick start

1. open V-REP and load the simulation scene in **./simulation_scene/test.ttt** 
2. start simulation in V-REP
3. run **test.py**

> The outcome of the simulation should be similar to the recorded video in **./video**

## Python Remote API

When use python as an external controller, follow the steps:

Add the following script to the child script of your robot model

```lua
simRemoteApi.start(19999)
```

Find **sim.py** **simConst.py** and **remoteApi.dll** (file extension depends on OS) and put them in the same folder as your script

An example structure is as following

```python
import sim

sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

   	# <your code here>

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
```

> For more info, please refer to the [official document]()

## Control UR10 in V-REP with Inverse kinematics

1. Open V-REP, select UR10 from model browser and put it into the scene

   <img src="./image/V-REP/1.png" alt="1" style="zoom:60%;" />

   

2. Set every joint to inverse kinematics mode

   The second button in the left tool bar is called "Scene object properties"

   <img src="./image/V-REP/2.png" alt="2" style="zoom:60%;" />

   Select a joint in the scene hierarchy section then click on the "Scene object properties" button to view and change the joint properties

   <img src="./image/V-REP/3.png" alt="3" style="zoom:80%;" />

3. For each link in the model, disable the dynamics

   <img src="./image/V-REP/4.png" alt="4" style="zoom:80%;" />

   **DO NOT FORGET the "UR10" at the top of the model hierarchy**

4. Link gripper to UR10

   Pick a gripper from model browser and put it into the scene (here I use RG2 gripper)

   Select the gripper, then **Ctrl-select** "UR10_connection" in UR10 model hierarchy

   <img src="./image/V-REP/5.png" alt="5" style="zoom:60%;" />

   Click the "assemble/disassemble" button in the top tool bar and it should automatically assemble

   ![6](./image/V-REP/6.png)

   > The gripper knew how to attach itself because it was appropriately configured during its model definition.
   >
   > More info about model assembling can be found in the last part of the [official tutorial](https://www.coppeliarobotics.com/helpFiles/en/buildingAModelTutorial.htm)

5. Determine the tip location (i.e. inverse kinematics tip)

   Add a dummy object where you think the tip should be and rename it to "UR10_tip"

   Dummy object can be added via [Menu bar -> Add -> Dummy]

   Then make the dummy object "UR10_tip" the child of "UR10_link7" by selecting "UR10_tip", then **Ctrl-select** "UR10_link7", and [Menu bar –> Edit –> Make last selected object parent]

   <img src="./image/V-REP/7.png" alt="7" style="zoom:80%;" />

   > The joint angle is changed in the figure above, you can change it in joint properties

6. Add a inverse kinematics target

   Copy and paste "UR10_tip", then rename the object to "UR10_target"

   <img src="./image/V-REP/8.png" alt="8" style="zoom:60%;" />

7. Create the "tip-target" pair

   Open the property dialog for "UR10_tip" and config it as the following figure

   ![9](./image/V-REP/9.png)

   After setting up the properties, you should see that "UR10_tip" and "UR10_target" are connected both in the hierarchy and the scene by red line marked in the figure above

   > If you only see one dummy object in the scene, it is probably because "UR10_tip" and "UR10_target" are overlapping. Use the "Object/item shift" button in the top tool bar to separate them

8. Create inverse kinematics group

   Now we only have to connected dummies, in this step we group them into a inverse kinematics group.

   Open the inverse kinematics dialog, add a new group and rename it to "UR10" and config it right as the following figure

   <img src="./image/V-REP/10.png" alt="10" style="zoom:60%;" />

   Then click "Edit IK elements" in the bottom of the dialog

   In the new dialog, select "UR10_tip" in the upper right menu, then click "Add new IK element with tip"

   <img src="./image/V-REP/11.png" alt="11" style="zoom:60%;" />

After clicking "Add new IK element with tip", "UR10_tip" should appear in the list. Then select it and config right as the figure above

> For detailed info about the parameters in the dialog, please refer to the following official documents:
>
> [Inverse kinematics dialog](https://www.coppeliarobotics.com/helpFiles/en/ikDialog.htm)
>
> [IK element dialog](https://www.coppeliarobotics.com/helpFiles/en/ikElementDialog.htm)

9. Test

   At this point the configuration should be done

   You can test it by running the simulation and use the "Object/item shift" button in the top tool bar to move "UR10_target". 

   UR10 are expected to follow the pose of "UR10_target"

   > Note that V-REP will try to fit the pose of "UR10_target" and "UR10_tip", so when testing manually you need to be careful if the position and orientation of "UR10_target" can be adapted by "UR10_tip"

* Control with python

  Link to the simulator sever (need the simulation running when executing the script)

  ```python
  sim.simxFinish(-1) # just in case, close all opened connections
  clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
  if clientID!=-1:
      print ('Connected to remote API server')
  
     	# <your code here>
  
      # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
      sim.simxGetPingTime(clientID)
  
      # Now close the connection to CoppeliaSim:
      sim.simxFinish(clientID)
  else:
      print ('Failed connecting to remote API server')
  print ('Program ended')
  ```

  Control "UR10_target" pose

  ```python
  sim_ret, target_handle = sim.simxGetObjectHandle(clientID, "UR10_target", sim.simx_opmode_blocking) # get object handle
  sim_ret, target_ori = sim.simxGetObjectOrientation(clientID, target_handle, -1, sim.simx_opmode_blocking) # get object orientation
  sim_ret, target_pos = sim.simxGetObjectPosition(clientID, target_handle, -1, sim.simx_opmode_blocking) # get object position
  
  # set target pose
  target_orientation = <your code here>
  target_position = <your code here>
  
  sim.simxSetObjectOrientation(clientID, target_handle, -1, target_orientation, sim.simx_opmode_blocking) # set object orientation
  sim.simxSetObjectPosition(clientID, target_handle, -1, target_position, sim.simx_opmode_blocking) # set object orientation
  ```

  

  ## Control UR10 with Joint Angle

  For directly control joints with Force/Torque mode, refer to [this](https://blog.csdn.net/weixin_41754912/article/details/82353012)

  

  ## Config and get camera parameter

  * Sensor work mode

    Perspective projection /  orthogonal projection mode can be configured in V-REP scene object properties [Vision sensor properties](https://www.coppeliarobotics.com/helpFiles/en/visionSensorPropertiesDialog.htm)

  * Camera intrinsics

    It seems that V-REP does not offer a API for camera intrinsics. In AndyZeng's repo, he directly use the following code to define the camera intrinsics

    ```python
    # robot.py line 151
    self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    ```

  * Camera extrinsics

    Camera extrinsics can be directly obtained via python API provided by V-REP according to AndyZeng's project

    ```python
    # robot.py line 142
    
    # Get camera pose and intrinsics in simulation
    sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
    sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
    cam_trans = np.eye(4,4)
    cam_trans[0:3,3] = np.asarray(cam_position)
    cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
    cam_rotm = np.eye(4,4)
    cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
    self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
    ```

    Camera position and orientation can be set in V-REP scene with the following steps:

    1. Right click on scene -> Add -> Vision Sensor -> Perspective type

       > Note that vision sensor is different with camera, more detailed info can be found [here](https://www.coppeliarobotics.com/helpFiles/en/visionSensors.htm)

    2. Select the vision sensor 

    3. Click the button "Object/item shift" on the upper tool bar. Then you can use the tool to adjust its pose

  