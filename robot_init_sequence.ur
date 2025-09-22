# Robot Initialization and Demo Sequence
# This script will work when robot is powered on and ready

# Simple movements that should work in the simulator
movej([0, -1.57, 0, -1.57, 0, 0], a=0.5, v=0.5)
sleep(1)
movej([0.5, -1.57, 0, -1.57, 0, 0], a=0.5, v=0.5)
sleep(1)
movej([0, -1.57, 0, -1.57, 0, 0], a=0.5, v=0.5)
sleep(1)