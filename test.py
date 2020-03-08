import sim

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    sim_ret, UR10_target_handle = sim.simxGetObjectHandle(clientID, "UR10_target", sim.simx_opmode_blocking)
    sim_ret, UR10_target_position = sim.simxGetObjectPosition(clientID, UR10_target_handle, -1, sim.simx_opmode_blocking)
    sim_ret, UR10_target_orientation = sim.simxGetObjectOrientation(clientID, UR10_target_handle, -1, sim.simx_opmode_blocking)

    move_z = 0.005
    rot_y = 0.01

    for step in range(50):
        UR10_target_position[2] += move_z
        UR10_target_orientation[1] += rot_y
        sim.simxSetObjectPosition(clientID, UR10_target_handle, -1, UR10_target_position, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(clientID, UR10_target_handle, -1, UR10_target_orientation, sim.simx_opmode_blocking)
    

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')