# Example URScript based on provided documentation
# This script demonstrates basic UR robot movements

# Define a known-good home position (joint space)
home = [-1.587, -1.587, -1.587, 0, 1.587, -3.1416]

# Move to home position using joint-space motion (most reliable)
movej(home, a=0.1, v=0.1)

# Get current position as a pose for relative movements
ref_pose = get_actual_tcp_pose()

# Demonstrate cartesian-space linear movements
# Move 10cm along Y axis
movel(pose_add(ref_pose, p[0, 0.1, 0, 0, 0, 0]), a=0.1, v=0.1)
sleep(0.5)

# Move back 20cm along Y axis  
movel(pose_add(ref_pose, p[0, -0.1, 0, 0, 0, 0]), a=0.1, v=0.1)
sleep(0.5)

# Move up 20cm along Z axis
movel(pose_add(ref_pose, p[0, 0, 0.2, 0, 0, 0]), a=0.1, v=0.1)
sleep(0.5)

# Move back down
movel(pose_add(ref_pose, p[0, 0, -0.2, 0, 0, 0]), a=0.1, v=0.1)
sleep(0.5)

# Demonstrate rotational movement around X axis (0.2 radians)
movel(pose_add(ref_pose, p[0, 0, 0, 0.2, 0, 0]), a=0.1, v=0.1)
sleep(0.5)

# Rotate back
movel(pose_add(ref_pose, p[0, 0, 0, -0.2, 0, 0]), a=0.1, v=0.1)
sleep(0.5)

# Return to home position using joint-space motion
movej(home, a=0.1, v=0.1)

# Display completion message
popup("Movement sequence completed!")