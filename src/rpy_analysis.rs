//! RPY Analysis Module
//!
//! Real-time analysis of Roll-Pitch-Yaw attitude effects on robot performance.
//! Based on the diffusion analysis originally developed in procrustesd/post_process.py
//! 
//! This module provides:
//! - Real-time RPY attitude tracking and correlation analysis
//! - Depth error analysis (when depth sensors are available)
//! - Statistical analysis of attitude vs performance metrics
//! - Configurable thresholds and analysis modes

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for RPY analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPYAnalysisConfig {
    /// Enable RPY analysis
    pub enabled: bool,
    /// Window size for statistical analysis (number of samples)
    pub analysis_window_size: usize,
    /// Correlation threshold for detecting significant relationships
    pub correlation_threshold: f64,
    /// Minimum samples before analysis starts
    pub min_samples: usize,
    /// Output analysis results every N samples
    pub output_interval: usize,
    /// Enable depth error analysis (requires depth sensor data)
    pub depth_analysis_enabled: bool,
    /// Maximum acceptable depth error percentage
    pub max_depth_error_percent: f64,
}

impl Default for RPYAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            analysis_window_size: 1000,
            correlation_threshold: 0.3,
            min_samples: 100,
            output_interval: 250,
            depth_analysis_enabled: false,
            max_depth_error_percent: 50.0,
        }
    }
}

/// RPY sample data point
#[derive(Debug, Clone)]
pub struct RPYSample {
    pub timestamp: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub yaw_rate_dps: f64,
    pub tcp_pose: [f64; 6],
    pub joint_positions: [f64; 6],
    // Optional depth/performance metrics
    pub depth_error_percent: Option<f64>,
    pub velocity_magnitude: Option<f64>,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPYStatistics {
    pub timestamp: f64,
    pub sample_count: usize,
    pub roll_stats: AttitudeStats,
    pub pitch_stats: AttitudeStats,
    pub yaw_rate_stats: AttitudeStats,
    pub correlations: RPYCorrelations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttitudeStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPYCorrelations {
    pub roll_vs_depth_error: Option<f64>,
    pub pitch_vs_depth_error: Option<f64>,
    pub yaw_rate_vs_depth_error: Option<f64>,
    pub roll_vs_velocity: Option<f64>,
    pub pitch_vs_velocity: Option<f64>,
    pub yaw_rate_vs_velocity: Option<f64>,
}

/// Real-time RPY analyzer
pub struct RPYAnalyzer {
    config: RPYAnalysisConfig,
    samples: VecDeque<RPYSample>,
    sample_count: usize,
    last_output: Option<Instant>,
    last_tcp_pose: Option<[f64; 6]>,
    last_timestamp: Option<f64>,
}

impl RPYAnalyzer {
    pub fn new(config: RPYAnalysisConfig) -> Self {
        Self {
            samples: VecDeque::with_capacity(config.analysis_window_size),
            config,
            sample_count: 0,
            last_output: None,
            last_tcp_pose: None,
            last_timestamp: None,
        }
    }

    /// Add a new RPY sample for analysis
    pub fn add_sample(
        &mut self,
        timestamp: f64,
        roll_deg: f64,
        pitch_deg: f64,
        yaw_rate_dps: f64,
        tcp_pose: [f64; 6],
        joint_positions: [f64; 6],
        depth_error_percent: Option<f64>,
    ) -> Option<RPYStatistics> {
        if !self.config.enabled {
            return None;
        }

        // Calculate velocity magnitude if we have previous data
        let velocity_magnitude = if let (Some(last_pose), Some(last_ts)) = 
            (self.last_tcp_pose, self.last_timestamp) {
            let dt = timestamp - last_ts;
            if dt > 0.0 {
                let dx = tcp_pose[0] - last_pose[0];
                let dy = tcp_pose[1] - last_pose[1];
                let dz = tcp_pose[2] - last_pose[2];
                let velocity = ((dx*dx + dy*dy + dz*dz).sqrt() / dt) * 1000.0; // mm/s
                Some(velocity)
            } else {
                None
            }
        } else {
            None
        };

        let sample = RPYSample {
            timestamp,
            roll_deg,
            pitch_deg,
            yaw_rate_dps,
            tcp_pose,
            joint_positions,
            depth_error_percent,
            velocity_magnitude,
        };

        // Add to sliding window
        if self.samples.len() >= self.config.analysis_window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        self.sample_count += 1;

        // Update state for next iteration
        self.last_tcp_pose = Some(tcp_pose);
        self.last_timestamp = Some(timestamp);

        // Check if we should output analysis
        if self.should_output_analysis() {
            self.last_output = Some(Instant::now());
            Some(self.compute_statistics())
        } else {
            None
        }
    }

    fn should_output_analysis(&self) -> bool {
        if self.samples.len() < self.config.min_samples {
            return false;
        }

        if self.sample_count % self.config.output_interval != 0 {
            return false;
        }

        true
    }

    fn compute_statistics(&self) -> RPYStatistics {
        let samples_vec: Vec<&RPYSample> = self.samples.iter().collect();
        
        let roll_stats = self.compute_attitude_stats(&samples_vec, |s| s.roll_deg);
        let pitch_stats = self.compute_attitude_stats(&samples_vec, |s| s.pitch_deg);
        let yaw_rate_stats = self.compute_attitude_stats(&samples_vec, |s| s.yaw_rate_dps);
        
        let correlations = self.compute_correlations(&samples_vec);

        RPYStatistics {
            timestamp: samples_vec.last().unwrap().timestamp,
            sample_count: samples_vec.len(),
            roll_stats,
            pitch_stats,
            yaw_rate_stats,
            correlations,
        }
    }

    fn compute_attitude_stats<F>(&self, samples: &[&RPYSample], extractor: F) -> AttitudeStats 
    where
        F: Fn(&RPYSample) -> f64,
    {
        let values: Vec<f64> = samples.iter().map(|s| extractor(s)).collect();
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        AttitudeStats {
            mean,
            std_dev,
            min,
            max,
            range,
        }
    }

    fn compute_correlations(&self, samples: &[&RPYSample]) -> RPYCorrelations {
        let _roll_values: Vec<f64> = samples.iter().map(|s| s.roll_deg).collect();
        let _pitch_values: Vec<f64> = samples.iter().map(|s| s.pitch_deg).collect();
        let _yaw_rate_values: Vec<f64> = samples.iter().map(|s| s.yaw_rate_dps).collect();
        
        // Depth error correlations (if available)
        let depth_errors: Vec<f64> = samples.iter()
            .filter_map(|s| s.depth_error_percent)
            .collect();
        
        let (roll_vs_depth_error, pitch_vs_depth_error, yaw_rate_vs_depth_error) = 
            if depth_errors.len() > 10 {
                let roll_depth: Vec<f64> = samples.iter()
                    .filter_map(|s| s.depth_error_percent.map(|_| s.roll_deg))
                    .collect();
                let pitch_depth: Vec<f64> = samples.iter()
                    .filter_map(|s| s.depth_error_percent.map(|_| s.pitch_deg))
                    .collect();
                let yaw_rate_depth: Vec<f64> = samples.iter()
                    .filter_map(|s| s.depth_error_percent.map(|_| s.yaw_rate_dps))
                    .collect();

                (
                    Some(self.pearson_correlation(&roll_depth, &depth_errors)),
                    Some(self.pearson_correlation(&pitch_depth, &depth_errors)),
                    Some(self.pearson_correlation(&yaw_rate_depth, &depth_errors)),
                )
            } else {
                (None, None, None)
            };

        // Velocity correlations (if available)
        let velocities: Vec<f64> = samples.iter()
            .filter_map(|s| s.velocity_magnitude)
            .collect();
        
        let (roll_vs_velocity, pitch_vs_velocity, yaw_rate_vs_velocity) = 
            if velocities.len() > 10 {
                let roll_vel: Vec<f64> = samples.iter()
                    .filter_map(|s| s.velocity_magnitude.map(|_| s.roll_deg))
                    .collect();
                let pitch_vel: Vec<f64> = samples.iter()
                    .filter_map(|s| s.velocity_magnitude.map(|_| s.pitch_deg))
                    .collect();
                let yaw_rate_vel: Vec<f64> = samples.iter()
                    .filter_map(|s| s.velocity_magnitude.map(|_| s.yaw_rate_dps))
                    .collect();

                (
                    Some(self.pearson_correlation(&roll_vel, &velocities)),
                    Some(self.pearson_correlation(&pitch_vel, &velocities)),
                    Some(self.pearson_correlation(&yaw_rate_vel, &velocities)),
                )
            } else {
                (None, None, None)
            };

        RPYCorrelations {
            roll_vs_depth_error,
            pitch_vs_depth_error,
            yaw_rate_vs_depth_error,
            roll_vs_velocity,
            pitch_vs_velocity,
            yaw_rate_vs_velocity,
        }
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Check if any correlation exceeds threshold (indicates significant relationship)
    pub fn has_significant_correlations(&self, stats: &RPYStatistics) -> bool {
        let threshold = self.config.correlation_threshold;
        
        [
            stats.correlations.roll_vs_depth_error,
            stats.correlations.pitch_vs_depth_error,
            stats.correlations.yaw_rate_vs_depth_error,
            stats.correlations.roll_vs_velocity,
            stats.correlations.pitch_vs_velocity,
            stats.correlations.yaw_rate_vs_velocity,
        ]
        .iter()
        .any(|&corr| corr.map_or(false, |c| c.abs() > threshold))
    }

    /// Get current configuration
    pub fn config(&self) -> &RPYAnalysisConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RPYAnalysisConfig) {
        // Resize buffer if window size changed
        if config.analysis_window_size != self.config.analysis_window_size {
            let mut new_samples = VecDeque::with_capacity(config.analysis_window_size);
            
            // Keep the most recent samples
            let keep_count = config.analysis_window_size.min(self.samples.len());
            for sample in self.samples.iter().rev().take(keep_count).rev() {
                new_samples.push_back(sample.clone());
            }
            
            self.samples = new_samples;
        }
        
        self.config = config;
    }
}

/// Output RPY statistics as JSON
pub fn output_rpy_statistics(stats: &RPYStatistics) {
    if let Ok(json) = serde_json::to_string(stats) {
        println!("{}", json);
    }
}

/// Helper function to compute yaw rate from orientation changes
pub fn compute_yaw_rate(
    current_orientation: [f64; 3], // [rx, ry, rz] in radians
    previous_orientation: [f64; 3],
    dt: f64, // time delta in seconds
) -> f64 {
    if dt <= 0.0 {
        return 0.0;
    }

    // Simple approximation using rz (yaw) component change
    let yaw_change = current_orientation[2] - previous_orientation[2];
    
    // Handle wrap-around for yaw angle
    let yaw_change_wrapped = if yaw_change > std::f64::consts::PI {
        yaw_change - 2.0 * std::f64::consts::PI
    } else if yaw_change < -std::f64::consts::PI {
        yaw_change + 2.0 * std::f64::consts::PI
    } else {
        yaw_change
    };
    
    // Convert to degrees per second
    (yaw_change_wrapped / dt).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpy_analyzer_basic() {
        let config = RPYAnalysisConfig {
            enabled: true,
            min_samples: 5,
            output_interval: 5,
            ..Default::default()
        };
        
        let mut analyzer = RPYAnalyzer::new(config);
        
        // Add samples
        for i in 0..10 {
            let timestamp = i as f64;
            let result = analyzer.add_sample(
                timestamp,
                (i as f64) * 0.1,      // roll
                (i as f64) * 0.05,     // pitch  
                (i as f64) * 2.0,      // yaw_rate
                [0.0; 6],              // tcp_pose
                [0.0; 6],              // joint_positions
                Some((i as f64) * 1.5), // depth_error_percent
            );
            
            if i == 4 || i == 9 {
                assert!(result.is_some());
            }
        }
    }

    #[test]
    fn test_correlation_calculation() {
        let config = RPYAnalysisConfig::default();
        let analyzer = RPYAnalyzer::new(config);
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = analyzer.pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001); // Perfect positive correlation
    }

    #[test] 
    fn test_yaw_rate_computation() {
        let current = [0.1, 0.2, 1.5];
        let previous = [0.1, 0.2, 1.0];
        let dt = 0.1;
        
        let yaw_rate = compute_yaw_rate(current, previous, dt);
        let expected = ((1.5 - 1.0) / dt).to_degrees();
        
        assert!((yaw_rate - expected).abs() < 0.001);
    }
}