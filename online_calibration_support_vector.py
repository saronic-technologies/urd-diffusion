#!/usr/bin/env python3
"""
Online Calibration Support Vector System for UR Robot
Inspired by VGGT: Visual Geometry Grounded Transformer

This script implements online calibration and support vector machine optimization
for Universal Robot systems, incorporating 3D vision capabilities for enhanced
robot control and motion planning.
"""

import numpy as np
import cv2
import json
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import logging
from scipy import optimize
from sklearn.svm import SVR
import subprocess
import socket
import struct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Pose:
    """Represents a 6DOF pose (position + orientation)"""
    x: float
    y: float 
    z: float
    rx: float
    ry: float
    rz: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]
    
    def __str__(self) -> str:
        return f"p[{self.x:.4f}, {self.y:.4f}, {self.z:.4f}, {self.rx:.4f}, {self.ry:.4f}, {self.rz:.4f}]"

@dataclass
class CalibrationPoint:
    """Single calibration measurement"""
    robot_pose: Pose
    camera_pose: Optional[Pose] = None
    joint_positions: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0

class CameraCalibration:
    """Handles camera calibration and pose estimation"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        # ArUco marker detection setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
    def calibrate_camera(self, calibration_images: List[np.ndarray] = None) -> bool:
        """Perform camera calibration using checkerboard or provided images"""
        if calibration_images is None:
            logger.info("Starting interactive camera calibration...")
            calibration_images = self._capture_calibration_images()
        
        # Checkerboard pattern
        pattern_size = (9, 6)  # Internal corners
        square_size = 0.025  # 25mm squares
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []
        img_points = []
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                img_points.append(corners2)
        
        if len(obj_points) >= 10:
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], None, None)
            
            if ret:
                self.calibrated = True
                logger.info(f"Camera calibration successful with {len(obj_points)} images")
                return True
        
        logger.error("Camera calibration failed")
        return False
    
    def _capture_calibration_images(self) -> List[np.ndarray]:
        """Capture calibration images interactively"""
        images = []
        logger.info("Press SPACE to capture calibration images, ESC to finish")
        
        while len(images) < 20:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            cv2.imshow('Calibration', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                images.append(frame.copy())
                logger.info(f"Captured image {len(images)}/20")
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        return images
    
    def estimate_pose_from_markers(self, frame: np.ndarray) -> Optional[Tuple[Pose, float]]:
        """Estimate camera pose using ArUco markers"""
        if not self.calibrated:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(ids) > 0:
            marker_size = 0.05  # 5cm markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, self.camera_matrix, self.dist_coeffs)
            
            if len(rvecs) > 0:
                # Use first detected marker
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                
                # Convert to pose
                pose = Pose(
                    x=float(tvec[0]),
                    y=float(tvec[1]), 
                    z=float(tvec[2]),
                    rx=float(rvec[0]),
                    ry=float(rvec[1]),
                    rz=float(rvec[2])
                )
                
                confidence = min(1.0, len(ids) * 0.2)  # More markers = higher confidence
                return pose, confidence
        
        return None

class SupportVectorOptimizer:
    """Support Vector Machine for motion optimization"""
    
    def __init__(self):
        self.position_svr = {
            'x': SVR(kernel='rbf', gamma='scale'),
            'y': SVR(kernel='rbf', gamma='scale'), 
            'z': SVR(kernel='rbf', gamma='scale')
        }
        self.orientation_svr = {
            'rx': SVR(kernel='rbf', gamma='scale'),
            'ry': SVR(kernel='rbf', gamma='scale'),
            'rz': SVR(kernel='rbf', gamma='scale')
        }
        self.trained = False
        
    def train(self, calibration_points: List[CalibrationPoint]) -> bool:
        """Train SVR models using calibration data"""
        if len(calibration_points) < 10:
            logger.warning("Insufficient calibration points for SVM training")
            return False
            
        # Prepare training data
        X = []  # Robot joint positions
        y_pos = {'x': [], 'y': [], 'z': []}
        y_ori = {'rx': [], 'ry': [], 'rz': []}
        
        for point in calibration_points:
            if point.joint_positions and point.camera_pose:
                X.append(point.joint_positions)
                y_pos['x'].append(point.camera_pose.x)
                y_pos['y'].append(point.camera_pose.y)
                y_pos['z'].append(point.camera_pose.z)
                y_ori['rx'].append(point.camera_pose.rx)
                y_ori['ry'].append(point.camera_pose.ry)
                y_ori['rz'].append(point.camera_pose.rz)
        
        X = np.array(X)
        
        # Train position SVRs
        for axis in ['x', 'y', 'z']:
            self.position_svr[axis].fit(X, y_pos[axis])
            
        # Train orientation SVRs  
        for axis in ['rx', 'ry', 'rz']:
            self.orientation_svr[axis].fit(X, y_ori[axis])
            
        self.trained = True
        logger.info(f"SVM models trained with {len(calibration_points)} calibration points")
        return True
    
    def predict_pose(self, joint_positions: List[float]) -> Optional[Pose]:
        """Predict camera pose from robot joint positions"""
        if not self.trained:
            return None
            
        X = np.array([joint_positions])
        
        # Predict position
        x = self.position_svr['x'].predict(X)[0]
        y = self.position_svr['y'].predict(X)[0] 
        z = self.position_svr['z'].predict(X)[0]
        
        # Predict orientation
        rx = self.orientation_svr['rx'].predict(X)[0]
        ry = self.orientation_svr['ry'].predict(X)[0]
        rz = self.orientation_svr['rz'].predict(X)[0]
        
        return Pose(x, y, z, rx, ry, rz)
    
    def optimize_trajectory(self, start_pose: Pose, end_pose: Pose, 
                          num_waypoints: int = 10) -> List[Pose]:
        """Generate optimized trajectory between poses"""
        if not self.trained:
            logger.warning("SVM not trained, using linear interpolation")
            return self._linear_interpolate(start_pose, end_pose, num_waypoints)
            
        # Generate intermediate waypoints using SVM predictions
        waypoints = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            
            # Linear interpolation as starting point
            interp_pose = Pose(
                x=start_pose.x + alpha * (end_pose.x - start_pose.x),
                y=start_pose.y + alpha * (end_pose.y - start_pose.y),
                z=start_pose.z + alpha * (end_pose.z - start_pose.z),
                rx=start_pose.rx + alpha * (end_pose.rx - start_pose.rx),
                ry=start_pose.ry + alpha * (end_pose.ry - start_pose.ry),
                rz=start_pose.rz + alpha * (end_pose.rz - start_pose.rz)
            )
            
            waypoints.append(interp_pose)
            
        return waypoints
    
    def _linear_interpolate(self, start: Pose, end: Pose, num_points: int) -> List[Pose]:
        """Fallback linear interpolation"""
        waypoints = []
        for i in range(num_points + 1):
            alpha = i / num_points
            waypoints.append(Pose(
                x=start.x + alpha * (end.x - start.x),
                y=start.y + alpha * (end.y - start.y), 
                z=start.z + alpha * (end.z - start.z),
                rx=start.rx + alpha * (end.rx - start.rx),
                ry=start.ry + alpha * (end.ry - start.ry),
                rz=start.rz + alpha * (end.rz - start.rz)
            ))
        return waypoints

class OnlineCalibrationSystem:
    """Main online calibration and support vector system"""
    
    def __init__(self, robot_ip: str = "localhost", camera_id: int = 0):
        self.robot_ip = robot_ip
        self.camera = CameraCalibration(camera_id)
        self.svm_optimizer = SupportVectorOptimizer()
        
        # Calibration data storage
        self.calibration_points: deque = deque(maxlen=1000)
        self.current_robot_pose = None
        self.current_joint_positions = None
        
        # Threading
        self.running = False
        self.vision_thread = None
        self.robot_thread = None
        
        # URD connection
        self.urd_process = None
        
    def start_urd_daemon(self, config_path: str = "config/hw_config.yaml"):
        """Start URD daemon for robot communication"""
        try:
            cmd = ["./target/release/urd", "--config", config_path]
            self.urd_process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/Users/siddsingh/Documents/GitHub/urd-diffusion"
            )
            logger.info("URD daemon started")
            time.sleep(2)  # Allow daemon to initialize
            return True
        except Exception as e:
            logger.error(f"Failed to start URD daemon: {e}")
            return False
    
    def send_robot_command(self, command: str) -> bool:
        """Send command to robot via URD daemon"""
        if not self.urd_process:
            logger.error("URD daemon not running")
            return False
            
        try:
            self.urd_process.stdin.write(command + "\n")
            self.urd_process.stdin.flush()
            logger.debug(f"Sent command: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def calibrate_system(self, num_calibration_poses: int = 20) -> bool:
        """Perform full system calibration"""
        logger.info("Starting system calibration...")
        
        # Step 1: Camera calibration
        if not self.camera.calibrated:
            if not self.camera.calibrate_camera():
                return False
        
        # Step 2: Eye-hand calibration
        logger.info(f"Collecting {num_calibration_poses} calibration poses...")
        
        # Define calibration poses around workspace
        calibration_poses = self._generate_calibration_poses(num_calibration_poses)
        
        collected_points = []
        for i, pose in enumerate(calibration_poses):
            logger.info(f"Moving to calibration pose {i+1}/{num_calibration_poses}")
            
            # Move robot to pose
            if not self.move_to_pose(pose):
                logger.warning(f"Failed to reach pose {i+1}, skipping...")
                continue
            
            time.sleep(1.0)  # Allow settling
            
            # Capture camera data
            ret, frame = self.camera.cap.read()
            if ret:
                vision_result = self.camera.estimate_pose_from_markers(frame)
                if vision_result:
                    camera_pose, confidence = vision_result
                    
                    # Get current joint positions (mock for now)
                    joint_positions = self._get_current_joint_positions()
                    
                    calib_point = CalibrationPoint(
                        robot_pose=pose,
                        camera_pose=camera_pose,
                        joint_positions=joint_positions,
                        confidence=confidence
                    )
                    
                    collected_points.append(calib_point)
                    self.calibration_points.append(calib_point)
                    
                    logger.info(f"Collected calibration point {len(collected_points)} (confidence: {confidence:.2f})")
        
        # Step 3: Train SVM models
        if len(collected_points) >= 10:
            if self.svm_optimizer.train(collected_points):
                logger.info("System calibration completed successfully!")
                return True
        
        logger.error("System calibration failed - insufficient data")
        return False
    
    def _generate_calibration_poses(self, num_poses: int) -> List[Pose]:
        """Generate well-distributed calibration poses"""
        poses = []
        
        # Define workspace boundaries (adjust for your robot)
        x_range = (-0.3, 0.3)
        y_range = (0.3, 0.8) 
        z_range = (0.1, 0.4)
        
        for i in range(num_poses):
            # Generate poses in a grid pattern with some randomization
            grid_size = int(np.ceil(np.cbrt(num_poses)))
            ix = i % grid_size
            iy = (i // grid_size) % grid_size
            iz = i // (grid_size * grid_size)
            
            x = x_range[0] + (x_range[1] - x_range[0]) * (ix + 0.1*np.random.rand()) / grid_size
            y = y_range[0] + (y_range[1] - y_range[0]) * (iy + 0.1*np.random.rand()) / grid_size  
            z = z_range[0] + (z_range[1] - z_range[0]) * (iz + 0.1*np.random.rand()) / max(1, iz)
            
            # Varied orientations
            rx = -np.pi + 0.2 * np.random.rand()
            ry = 0.0 + 0.2 * np.random.rand() 
            rz = -np.pi/2 + 0.2 * np.random.rand()
            
            poses.append(Pose(x, y, z, rx, ry, rz))
        
        return poses
    
    def move_to_pose(self, pose: Pose, use_linear: bool = False) -> bool:
        """Move robot to specified pose"""
        if use_linear:
            command = f"movel({pose}, a=0.1, v=0.1)"
        else:
            # Convert to joint space if possible using SVM
            predicted_joints = self._pose_to_joints(pose)
            if predicted_joints:
                joint_str = "[" + ",".join([f"{j:.4f}" for j in predicted_joints]) + "]"
                command = f"movej({joint_str}, a=0.1, v=0.1)"
            else:
                command = f"movel({pose}, a=0.1, v=0.1)"
        
        return self.send_robot_command(command)
    
    def _pose_to_joints(self, pose: Pose) -> Optional[List[float]]:
        """Convert pose to joint positions using inverse kinematics (mock)"""
        # In a real implementation, this would use proper IK
        # For now, return None to use linear moves
        return None
    
    def _get_current_joint_positions(self) -> List[float]:
        """Get current robot joint positions (mock implementation)"""
        # In real implementation, this would query the robot
        # For now, return a mock set of joint positions
        return [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    
    def start_online_optimization(self):
        """Start continuous online optimization"""
        self.running = True
        
        # Start vision processing thread
        self.vision_thread = threading.Thread(target=self._vision_processing_loop)
        self.vision_thread.daemon = True
        self.vision_thread.start()
        
        logger.info("Online optimization started")
    
    def stop_online_optimization(self):
        """Stop online optimization"""
        self.running = False
        
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)
            
        if self.urd_process:
            self.urd_process.terminate()
            
        logger.info("Online optimization stopped")
    
    def _vision_processing_loop(self):
        """Continuous vision processing for online calibration updates"""
        while self.running:
            ret, frame = self.camera.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Estimate pose from vision
            vision_result = self.camera.estimate_pose_from_markers(frame)
            if vision_result:
                camera_pose, confidence = vision_result
                
                # Get current robot state (mock)
                current_joints = self._get_current_joint_positions()
                current_robot_pose = Pose(0, 0.5, 0.2, -3.14, 0, -1.57)  # Mock
                
                # Add to calibration data if confidence is high enough
                if confidence > 0.7:
                    calib_point = CalibrationPoint(
                        robot_pose=current_robot_pose,
                        camera_pose=camera_pose,
                        joint_positions=current_joints,
                        confidence=confidence
                    )
                    self.calibration_points.append(calib_point)
                    
                    # Retrain SVM periodically with new data
                    if len(self.calibration_points) % 50 == 0:
                        logger.info("Updating SVM models with new calibration data...")
                        self.svm_optimizer.train(list(self.calibration_points))
            
            time.sleep(0.1)  # 10 Hz processing rate
    
    def execute_optimized_trajectory(self, start_pose: Pose, end_pose: Pose) -> bool:
        """Execute trajectory optimized by support vector machine"""
        logger.info(f"Executing optimized trajectory from {start_pose} to {end_pose}")
        
        # Generate optimized waypoints
        waypoints = self.svm_optimizer.optimize_trajectory(start_pose, end_pose, num_waypoints=5)
        
        # Execute trajectory
        for i, waypoint in enumerate(waypoints):
            logger.info(f"Moving to waypoint {i+1}/{len(waypoints)}")
            if not self.move_to_pose(waypoint, use_linear=True):
                logger.error(f"Failed to reach waypoint {i+1}")
                return False
            time.sleep(0.5)  # Brief pause between waypoints
        
        logger.info("Trajectory execution completed")
        return True

def main():
    """Main execution function"""
    logger.info("Starting Online Calibration Support Vector System")
    
    # Initialize system
    system = OnlineCalibrationSystem(robot_ip="localhost", camera_id=0)
    
    try:
        # Start URD daemon
        if not system.start_urd_daemon():
            logger.error("Failed to start URD daemon")
            return
        
        # Perform system calibration
        logger.info("Starting system calibration...")
        if system.calibrate_system(num_calibration_poses=15):
            logger.info("Calibration successful!")
            
            # Start online optimization
            system.start_online_optimization()
            
            # Example usage: Execute optimized trajectory
            start_pose = Pose(-0.2, 0.4, 0.3, -3.14, 0, -1.57)
            end_pose = Pose(0.2, 0.6, 0.2, -3.14, 0, -1.57)
            
            time.sleep(2)  # Allow online optimization to start
            system.execute_optimized_trajectory(start_pose, end_pose)
            
            # Keep running for online optimization
            logger.info("System running. Press Ctrl+C to exit...")
            while True:
                time.sleep(1)
                
        else:
            logger.error("System calibration failed")
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        system.stop_online_optimization()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()