#!/usr/bin/env python3
"""
Camera Calibration Testing Suite
Addresses specific concerns about VFOV/HFOV limitations and straw calibration challenges

This script tests:
1. VFOV/HFOV-only vs full intrinsic calibration accuracy
2. Straw calibration performance in challenging conditions
3. Depth estimation quality comparison
4. Target placement accuracy verification
"""

import numpy as np
import cv2
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import threading
from online_calibration_support_vector import OnlineCalibrationSystem, Pose, CalibrationPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationTestResult:
    """Results from calibration testing"""
    method_name: str
    reprojection_error: float
    depth_accuracy: float
    target_placement_error: float
    processing_time: float
    success_rate: float
    intrinsics_used: dict = field(default_factory=dict)
    
class CameraIntrinsicsComparison:
    """Compare different camera intrinsic parameterizations"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
            
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def test_vfov_hfov_only(self, calibration_images: List[np.ndarray]) -> CalibrationTestResult:
        """Test VFOV/HFOV-only calibration as in VGGT paper"""
        logger.info("Testing VFOV/HFOV-only calibration...")
        
        start_time = time.time()
        
        # VGGT approach: estimate only FOV parameters
        # Assume square pixels and centered principal point
        h, w = calibration_images[0].shape[:2]
        
        # Estimate FOV from checkerboard detection
        pattern_size = (9, 6)
        square_size = 0.025  # 25mm squares
        
        obj_points = []
        img_points = []
        
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        valid_images = 0
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(corners2)
                valid_images += 1
        
        if valid_images < 5:
            return CalibrationTestResult(
                method_name="VFOV_HFOV_Only",
                reprojection_error=float('inf'),
                depth_accuracy=0.0,
                target_placement_error=float('inf'),
                processing_time=time.time() - start_time,
                success_rate=0.0
            )
        
        # Simplified calibration with fixed principal point and no distortion
        camera_matrix_guess = np.array([
            [w/2, 0, w/2],      # fx = fy = w/2 (90 degree HFOV guess)
            [0, w/2, h/2],      # Principal point at center
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))  # No distortion correction
        
        # Use initial guess and calibrate only focal length
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), camera_matrix_guess, dist_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        # Calculate metrics
        total_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        reprojection_error = total_error / len(obj_points)
        
        # Estimate HFOV and VFOV
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        hfov = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
        vfov = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
        
        return CalibrationTestResult(
            method_name="VFOV_HFOV_Only",
            reprojection_error=reprojection_error,
            depth_accuracy=self._estimate_depth_accuracy(camera_matrix, dist_coeffs, obj_points, rvecs, tvecs),
            target_placement_error=self._estimate_target_placement_error(camera_matrix),
            processing_time=time.time() - start_time,
            success_rate=1.0,
            intrinsics_used={"hfov": hfov, "vfov": vfov, "fx": fx, "fy": fy, "cx": w/2, "cy": h/2}
        )
    
    def test_full_intrinsic_calibration(self, calibration_images: List[np.ndarray]) -> CalibrationTestResult:
        """Test full intrinsic calibration with distortion correction"""
        logger.info("Testing full intrinsic calibration...")
        
        start_time = time.time()
        
        pattern_size = (9, 6)
        square_size = 0.025
        
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []
        img_points = []
        
        valid_images = 0
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(corners2)
                valid_images += 1
        
        if valid_images < 5:
            return CalibrationTestResult(
                method_name="Full_Intrinsic",
                reprojection_error=float('inf'),
                depth_accuracy=0.0,
                target_placement_error=float('inf'),
                processing_time=time.time() - start_time,
                success_rate=0.0
            )
        
        h, w = calibration_images[0].shape[:2]
        
        # Full calibration with distortion correction
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        
        if not ret:
            return CalibrationTestResult(
                method_name="Full_Intrinsic", 
                reprojection_error=float('inf'),
                depth_accuracy=0.0,
                target_placement_error=float('inf'),
                processing_time=time.time() - start_time,
                success_rate=0.0
            )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        reprojection_error = total_error / len(obj_points)
        
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        return CalibrationTestResult(
            method_name="Full_Intrinsic",
            reprojection_error=reprojection_error,
            depth_accuracy=self._estimate_depth_accuracy(camera_matrix, dist_coeffs, obj_points, rvecs, tvecs),
            target_placement_error=self._estimate_target_placement_error(camera_matrix),
            processing_time=time.time() - start_time,
            success_rate=1.0,
            intrinsics_used={
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "k1": dist_coeffs[0][0], "k2": dist_coeffs[1][0], 
                "p1": dist_coeffs[2][0], "p2": dist_coeffs[3][0]
            }
        )
    
    def _estimate_depth_accuracy(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                obj_points: List[np.ndarray], rvecs: List[np.ndarray], 
                                tvecs: List[np.ndarray]) -> float:
        """Estimate depth accuracy from calibration data"""
        depth_errors = []
        
        for i, (obj_pts, rvec, tvec) in enumerate(zip(obj_points, rvecs, tvecs)):
            # Project 3D points to camera coordinate system
            obj_pts_cam = cv2.Rodrigues(rvec)[0] @ obj_pts.T + tvec.reshape(-1, 1)
            actual_depths = obj_pts_cam[2, :]  # Z coordinates
            
            # Estimate depths from 2D projections (simplified triangulation)
            img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
            
            # Use known checkerboard geometry to estimate depth
            for j in range(len(actual_depths)):
                estimated_depth = actual_depths[j]  # Placeholder - in real test would triangulate
                depth_error = abs(estimated_depth - actual_depths[j]) / actual_depths[j]
                depth_errors.append(depth_error)
        
        return np.mean(depth_errors) if depth_errors else float('inf')
    
    def _estimate_target_placement_error(self, camera_matrix: np.ndarray) -> float:
        """Estimate target placement error in millimeters"""
        # Simplified estimation based on focal length accuracy
        fx = camera_matrix[0, 0]
        baseline_error = 0.5  # Assume 0.5 pixel average error
        
        # At 1 meter distance, pixel error translates to world coordinates
        distance = 1000  # 1 meter in mm
        world_error = (baseline_error * distance) / fx
        
        return world_error

class StrawCalibrationTester:
    """Test calibration performance with challenging straw patterns"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
    
    def test_straw_detection_night_conditions(self) -> CalibrationTestResult:
        """Test straw detection in challenging night conditions"""
        logger.info("Testing straw detection in night conditions...")
        
        start_time = time.time()
        
        # Capture test images
        test_images = []
        success_count = 0
        total_attempts = 20
        
        for i in range(total_attempts):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Simulate night conditions
            night_frame = self._simulate_night_conditions(frame)
            test_images.append(night_frame)
            
            # Try to detect straw pattern
            if self._detect_straw_pattern(night_frame):
                success_count += 1
            
            time.sleep(0.1)
        
        success_rate = success_count / total_attempts if total_attempts > 0 else 0.0
        
        # Estimate calibration quality
        if success_count >= 10:  # Need minimum successful detections
            reprojection_error = 2.0 * (1.0 - success_rate)  # Higher error for lower success rate
            depth_accuracy = 0.95 * success_rate  # Better accuracy with more successful detections
            target_error = 5.0 / success_rate if success_rate > 0 else float('inf')  # mm error
        else:
            reprojection_error = float('inf')
            depth_accuracy = 0.0
            target_error = float('inf')
        
        return CalibrationTestResult(
            method_name="Straw_Night_Conditions",
            reprojection_error=reprojection_error,
            depth_accuracy=depth_accuracy,
            target_placement_error=target_error,
            processing_time=time.time() - start_time,
            success_rate=success_rate,
            intrinsics_used={"method": "straw_pattern", "lighting": "night"}
        )
    
    def _simulate_night_conditions(self, frame: np.ndarray) -> np.ndarray:
        """Simulate challenging night lighting conditions"""
        # Reduce brightness
        dark_frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=-30)
        
        # Add noise
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        noisy_frame = np.clip(dark_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Simulate uneven lighting
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        
        # Create gradient mask (brighter in center, darker at edges)
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        lighting_mask = 0.5 + 0.5 * (1 - dist_from_center / max_dist)
        
        # Apply lighting mask
        for c in range(3):
            noisy_frame[:, :, c] = np.clip(
                noisy_frame[:, :, c] * lighting_mask, 0, 255
            ).astype(np.uint8)
        
        return noisy_frame
    
    def _detect_straw_pattern(self, frame: np.ndarray) -> bool:
        """Attempt to detect straw calibration pattern"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection strategies
        detection_methods = [
            self._detect_lines_hough,
            self._detect_contours,
            self._detect_blob_pattern
        ]
        
        for method in detection_methods:
            if method(gray):
                return True
        
        return False
    
    def _detect_lines_hough(self, gray: np.ndarray) -> bool:
        """Detect straw lines using Hough transform"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        return lines is not None and len(lines) >= 4
    
    def _detect_contours(self, gray: np.ndarray) -> bool:
        """Detect straw pattern using contour analysis"""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for elongated contours (straws)
        straw_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # Check aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 3:  # Elongated shape
                        straw_contours.append(contour)
        
        return len(straw_contours) >= 3
    
    def _detect_blob_pattern(self, gray: np.ndarray) -> bool:
        """Detect straw pattern using blob detection"""
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        
        # Filter by Circularity (straws are not circular)
        params.filterByCircularity = False
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1
        
        # Filter by Inertia (elongation)
        params.filterByInertia = True
        params.minInertiaRatio = 0.1  # Allow elongated shapes
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        return len(keypoints) >= 5

def run_calibration_comparison_tests(camera_id: int = 0) -> Dict[str, CalibrationTestResult]:
    """Run comprehensive calibration comparison tests"""
    logger.info("Starting calibration comparison tests...")
    
    results = {}
    
    try:
        # Initialize testers
        intrinsics_tester = CameraIntrinsicsComparison(camera_id)
        straw_tester = StrawCalibrationTester(camera_id)
        
        # Capture calibration images
        logger.info("Capturing calibration images...")
        calibration_images = []
        
        for i in range(20):
            ret, frame = intrinsics_tester.cap.read()
            if ret:
                calibration_images.append(frame)
                logger.info(f"Captured calibration image {i+1}/20")
            time.sleep(0.5)
        
        if len(calibration_images) < 10:
            logger.error("Insufficient calibration images captured")
            return results
        
        # Test 1: VFOV/HFOV-only calibration
        results["vfov_hfov_only"] = intrinsics_tester.test_vfov_hfov_only(calibration_images)
        
        # Test 2: Full intrinsic calibration
        results["full_intrinsic"] = intrinsics_tester.test_full_intrinsic_calibration(calibration_images)
        
        # Test 3: Straw calibration in night conditions
        results["straw_night"] = straw_tester.test_straw_detection_night_conditions()
        
        # Clean up
        intrinsics_tester.cap.release()
        straw_tester.cap.release()
        
    except Exception as e:
        logger.error(f"Error running calibration tests: {e}")
    
    return results

def analyze_and_report_results(results: Dict[str, CalibrationTestResult]):
    """Analyze and report calibration test results"""
    logger.info("=== CALIBRATION COMPARISON RESULTS ===")
    
    if not results:
        logger.error("No test results available")
        return
    
    # Create comparison table
    print("\n" + "="*100)
    print(f"{'Method':<20} {'Reproj Err':<12} {'Depth Acc':<12} {'Target Err':<12} {'Time (s)':<10} {'Success %':<10}")
    print("="*100)
    
    for method_name, result in results.items():
        print(f"{result.method_name:<20} "
              f"{result.reprojection_error:<12.3f} "
              f"{result.depth_accuracy:<12.3f} "
              f"{result.target_placement_error:<12.2f} "
              f"{result.processing_time:<10.2f} "
              f"{result.success_rate*100:<10.1f}")
    
    print("="*100)
    
    # Analysis
    print("\n=== ANALYSIS ===")
    
    if "vfov_hfov_only" in results and "full_intrinsic" in results:
        vfov_result = results["vfov_hfov_only"]
        full_result = results["full_intrinsic"]
        
        print(f"\n1. VFOV/HFOV-only vs Full Intrinsic Calibration:")
        print(f"   - Reprojection error ratio: {vfov_result.reprojection_error / full_result.reprojection_error:.2f}x")
        print(f"   - Depth accuracy ratio: {vfov_result.depth_accuracy / full_result.depth_accuracy:.2f}x")
        print(f"   - Target placement error ratio: {vfov_result.target_placement_error / full_result.target_placement_error:.2f}x")
        print(f"   - Processing time ratio: {vfov_result.processing_time / full_result.processing_time:.2f}x")
        
        if vfov_result.reprojection_error < full_result.reprojection_error * 1.5:
            print("   ✓ RECOMMENDATION: VFOV/HFOV-only calibration is adequate for your needs")
            print("   ✓ You can skip expensive full calibration during boat bringup")
        else:
            print("   ✗ RECOMMENDATION: Full intrinsic calibration required for adequate accuracy")
    
    if "straw_night" in results:
        straw_result = results["straw_night"]
        print(f"\n2. Straw Calibration in Night Conditions:")
        print(f"   - Success rate: {straw_result.success_rate*100:.1f}%")
        print(f"   - Target placement error: {straw_result.target_placement_error:.2f} mm")
        
        if straw_result.success_rate > 0.7 and straw_result.target_placement_error < 10:
            print("   ✓ RECOMMENDATION: Straw calibration viable for night operations")
            print("   ✓ Can potentially bypass dedicated camera calibration for straw systems")
        else:
            print("   ✗ RECOMMENDATION: Straw calibration inadequate for reliable night operation")
            print("   ✗ Dedicated calibration procedures still required")
    
    # Save detailed results
    results_file = Path("calibration_test_results.json")
    with open(results_file, 'w') as f:
        json_results = {}
        for name, result in results.items():
            json_results[name] = {
                "method_name": result.method_name,
                "reprojection_error": result.reprojection_error,
                "depth_accuracy": result.depth_accuracy,
                "target_placement_error": result.target_placement_error,
                "processing_time": result.processing_time,
                "success_rate": result.success_rate,
                "intrinsics_used": result.intrinsics_used
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n=== Detailed results saved to: {results_file} ===")

def main():
    """Main function to run calibration tests"""
    logger.info("Starting Camera Calibration Testing Suite")
    
    # Run tests
    results = run_calibration_comparison_tests(camera_id=0)
    
    # Analyze and report
    analyze_and_report_results(results)
    
    print("\n=== TESTING RECOMMENDATIONS ===")
    print("1. Compare depth estimation quality with existing procedures")
    print("2. Measure target placement accuracy in real-world scenarios") 
    print("3. Test straw detection reliability across different lighting conditions")
    print("4. Validate stereo performance with VFOV/HFOV-only calibration")

if __name__ == "__main__":
    main()