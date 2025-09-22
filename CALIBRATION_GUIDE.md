# Online Calibration Support Vector System

A comprehensive system for online robot calibration and motion optimization using Support Vector Machines, inspired by the VGGT (Visual Geometry Grounded Transformer) approach.

## Overview

This system provides:
- **Real-time camera calibration** with ArUco markers
- **Eye-hand calibration** between camera and robot coordinate systems  
- **Support Vector Machine optimization** for trajectory planning
- **Online learning** that continuously improves calibration accuracy
- **Integration with URD daemon** for Universal Robots control

## Quick Start

### 1. Prerequisites

- Python 3.7+
- Universal Robot (physical or simulated)
- USB camera
- ArUco markers (5cm, DICT_6X6_250)
- Checkerboard pattern (9x6 internal corners, 25mm squares)

### 2. Installation

```bash
# Clone or ensure you're in the URD directory
cd /Users/siddsingh/Documents/GitHub/urd-diffusion

# Install Python dependencies
pip install -r requirements.txt

# Build URD daemon (if not already built)
cargo build --release
```

### 3. Basic Usage

```bash
# Launch with default settings (camera 0, localhost robot)
./launch_calibration_system.sh

# Use specific camera and robot IP
./launch_calibration_system.sh -c 1 -r 192.168.1.100

# Run in simulation mode
./launch_calibration_system.sh -s

# Skip initial calibration (use existing models)
./launch_calibration_system.sh -n
```

## System Components

### 1. Camera Calibration (`CameraCalibration`)

Handles intrinsic camera calibration using:
- **Checkerboard method**: Classic calibration with known pattern
- **ArUco marker detection**: Real-time pose estimation
- **Automatic quality assessment**: Filters low-quality measurements

**Key Features:**
- Interactive calibration image capture
- Sub-pixel corner refinement
- Distortion correction
- Real-time pose tracking

### 2. Support Vector Machine Optimizer (`SupportVectorOptimizer`)

Uses SVR (Support Vector Regression) for:
- **Joint-to-Cartesian mapping**: Predicts end-effector pose from joint angles
- **Trajectory optimization**: Generates smooth, optimal paths
- **Online learning**: Continuously improves with new data

**Benefits:**
- Non-linear relationship modeling
- Robust to outliers
- Handles high-dimensional joint space
- Real-time prediction

### 3. Online Calibration System (`OnlineCalibrationSystem`)

Main system coordinator that:
- Manages robot-camera calibration process
- Coordinates data collection and model training
- Provides real-time optimization
- Handles error recovery and safety

## Calibration Process

### Phase 1: Camera Calibration

1. **Setup**: Position checkerboard in robot workspace
2. **Capture**: System guides you through capturing 15+ calibration images
3. **Processing**: Automatic corner detection and parameter estimation
4. **Validation**: Quality assessment and distortion correction

### Phase 2: Eye-Hand Calibration

1. **Pose Generation**: System generates well-distributed calibration poses
2. **Data Collection**: Robot moves to each pose, camera observes ArUco markers
3. **Correspondence**: Links robot poses to camera observations
4. **Quality Filtering**: Removes low-confidence measurements

### Phase 3: SVM Training

1. **Feature Preparation**: Joint positions as input, camera poses as output
2. **Model Training**: Separate SVR models for position (x,y,z) and orientation (rx,ry,rz)
3. **Cross-Validation**: Ensures model generalization
4. **Performance Assessment**: Reports calibration accuracy

## Configuration

### Main Configuration (`config/calibration_config.yaml`)

```yaml
# Camera settings
camera:
  device_id: 0
  resolution: {width: 1280, height: 720}
  checkerboard:
    pattern_size: [9, 6]
    square_size: 0.025

# Robot workspace
robot:
  workspace:
    x_min: -0.4
    x_max: 0.4
    y_min: 0.2
    y_max: 0.8
    z_min: 0.05
    z_max: 0.5

# SVM parameters
svm:
  kernel: "rbf"
  gamma: "scale"
  C: 1.0
```

### URD Configuration (`config/hw_config.yaml`)

Standard URD configuration for robot connection.

## Usage Examples

### Example 1: Basic Calibration

```python
from online_calibration_support_vector import OnlineCalibrationSystem

# Initialize system
system = OnlineCalibrationSystem(robot_ip="192.168.1.100", camera_id=0)

# Start URD daemon
system.start_urd_daemon("config/hw_config.yaml")

# Perform calibration
system.calibrate_system(num_calibration_poses=20)

# Start online optimization
system.start_online_optimization()
```

### Example 2: Optimized Trajectory Execution

```python
from online_calibration_support_vector import Pose

# Define start and end poses
start_pose = Pose(-0.2, 0.4, 0.3, -3.14, 0, -1.57)
end_pose = Pose(0.2, 0.6, 0.2, -3.14, 0, -1.57)

# Execute optimized trajectory
system.execute_optimized_trajectory(start_pose, end_pose)
```

### Example 3: Real-time Pose Estimation

```python
# Get real-time camera pose estimation
ret, frame = system.camera.cap.read()
if ret:
    result = system.camera.estimate_pose_from_markers(frame)
    if result:
        pose, confidence = result
        print(f"Camera pose: {pose} (confidence: {confidence:.2f})")
```

## Advanced Features

### 1. Online Learning

The system continuously improves calibration accuracy:

```python
# Automatic background learning
system.start_online_optimization()

# Manual data point addition
calibration_point = CalibrationPoint(
    robot_pose=current_robot_pose,
    camera_pose=estimated_camera_pose,
    joint_positions=current_joints,
    confidence=0.95
)
system.calibration_points.append(calibration_point)
```

### 2. Trajectory Optimization

Generate smooth trajectories using SVM:

```python
# Generate optimized waypoints
waypoints = system.svm_optimizer.optimize_trajectory(
    start_pose, end_pose, num_waypoints=10
)

# Execute with custom timing
for waypoint in waypoints:
    system.move_to_pose(waypoint)
    time.sleep(0.5)
```

### 3. Quality Assessment

Monitor calibration quality:

```python
# Check calibration accuracy
accuracy_metrics = system.assess_calibration_quality()
print(f"Mean reprojection error: {accuracy_metrics['mean_error']:.2f} pixels")
print(f"Calibration confidence: {accuracy_metrics['confidence']:.2f}")
```

## Troubleshooting

### Common Issues

**1. Camera not detected**
```bash
# Check camera permissions
ls -la /dev/video*

# Test camera access
python3 -c "import cv2; print(cv2.VideoCapture(0).read()[0])"
```

**2. Robot connection failed**
```bash
# Verify robot IP
ping 192.168.1.100

# Check URD daemon
./target/release/urd --config config/hw_config.yaml
```

**3. ArUco markers not detected**
- Ensure proper lighting
- Check marker size configuration
- Verify camera focus
- Print high-quality markers

**4. Poor calibration accuracy**
- Collect more calibration poses
- Improve marker visibility
- Check camera stability
- Verify workspace boundaries

### Debug Mode

Enable verbose logging:

```bash
./launch_calibration_system.sh --verbose
```

Check log files:
```bash
tail -f logs/calibration_*.log
```

## API Reference

### Core Classes

#### `Pose`
Represents a 6DOF pose (position + orientation).
```python
pose = Pose(x=0.1, y=0.2, z=0.3, rx=0.0, ry=0.0, rz=1.57)
print(pose)  # p[0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 1.5700]
```

#### `CalibrationPoint`
Single calibration measurement linking robot and camera poses.

#### `CameraCalibration`
Handles camera calibration and pose estimation.

#### `SupportVectorOptimizer`
SVM-based trajectory optimization and pose prediction.

#### `OnlineCalibrationSystem`
Main system coordinator.

### Key Methods

#### `calibrate_system(num_calibration_poses=20)`
Performs complete system calibration.

#### `start_online_optimization()`
Begins continuous online learning and optimization.

#### `execute_optimized_trajectory(start_pose, end_pose)`
Executes SVM-optimized trajectory between poses.

#### `move_to_pose(pose, use_linear=False)`
Moves robot to specified pose using optimal motion type.

## Performance Tips

### 1. Calibration Quality

- Use **20+ calibration poses** for good accuracy
- Ensure **even distribution** across workspace
- Maintain **good lighting** for marker detection
- Use **high-quality printed markers**

### 2. Real-time Performance

- Set appropriate **processing rates** (10 Hz for vision)
- Use **background threading** for continuous learning
- Enable **incremental SVM updates** for efficiency
- Monitor **system resource usage**

### 3. Motion Optimization

- Train SVM with **diverse training data**
- Use **appropriate kernel parameters**
- Enable **cross-validation** for parameter tuning
- Monitor **prediction accuracy**

## Integration with URD

The system integrates seamlessly with URD:

### URScript Command Generation

```python
# Generate URScript commands
command = f"movel({pose}, a=0.1, v=0.1)"
system.send_robot_command(command)

# Joint space motion
joint_str = "[" + ",".join([f"{j:.4f}" for j in joints]) + "]"
command = f"movej({joint_str}, a=0.1, v=0.1)"
```

### Real-time Monitoring

```python
# Monitor robot state via URD RTDE
current_pose = system._get_current_pose()
current_joints = system._get_current_joint_positions()
```

## Future Enhancements

### VGGT Integration

Future versions may include:
- **Transformer-based pose estimation**
- **Multi-view 3D reconstruction**
- **Dense point cloud processing**
- **End-to-end learning pipelines**

### Advanced Features

- **Collision avoidance** with SVM optimization
- **Dynamic obstacle handling**
- **Multi-robot coordination**
- **Learning from demonstration**

## Safety Considerations

⚠️ **Important Safety Notes:**

1. **Always verify** robot workspace boundaries
2. **Use emergency stop** functionality when testing
3. **Monitor robot motion** during calibration
4. **Validate trajectories** before execution
5. **Keep clear** of robot workspace during operation

## Support and Contribution

For issues, questions, or contributions:

1. Check the troubleshooting section
2. Review log files for error details
3. Verify configuration settings
4. Test with simulation mode first

## License

This system builds upon the URD (Universal Robots Daemon) project and incorporates concepts from VGGT research. Please respect all applicable licenses and safety requirements when using with physical robot systems.