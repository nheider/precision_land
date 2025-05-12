#!/usr/bin/env python

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus

from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math
import time

class LinearPrecisionLanding(Node):
    def __init__(self):
        super().__init__('linear_precision_landing')
        
        # QoS settings for PX4
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        
        # PX4 publishers/subscribers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 
            '/fmu/in/offboard_control_mode', 
            qos_profile)
            
        self.publisher_trajectory = self.create_publisher(
            TrajectorySetpoint, 
            '/fmu/in/trajectory_setpoint', 
            qos_profile)
            
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile)
        
        # TF listener for AprilTag pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer for control loop
        timer_period = 0.05  # 20Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)
        
        # Parameters
        self.declare_parameter('tag_frame', 'base')
        self.declare_parameter('camera_frame', 'x500_mono_cam_down_0/camera_link/imager')
        self.declare_parameter('search_altitude', 2.0)
        self.declare_parameter('hover_altitude', 1.0)
        self.declare_parameter('descent_rate', 0.1)  # m/s
        self.declare_parameter('search_x', 0.0)
        self.declare_parameter('search_y', 0.0)
        self.declare_parameter('xy_tolerance', 0.1)  # 10cm position tolerance
        self.declare_parameter('detection_timeout', 1.0)  # Time in seconds to consider tag lost
        self.declare_parameter('search_pattern_enabled', True)
        self.declare_parameter('search_pattern_radius', 2.0)
        self.declare_parameter('search_pattern_speed', 0.5)
        
        # Load parameters
        self.tag_frame = self.get_parameter('tag_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.search_altitude = self.get_parameter('search_altitude').value
        self.hover_altitude = self.get_parameter('hover_altitude').value
        self.descent_rate = self.get_parameter('descent_rate').value
        self.search_x = self.get_parameter('search_x').value
        self.search_y = self.get_parameter('search_y').value
        self.xy_tolerance = self.get_parameter('xy_tolerance').value
        self.detection_timeout = self.get_parameter('detection_timeout').value
        self.search_pattern_enabled = self.get_parameter('search_pattern_enabled').value
        self.search_pattern_radius = self.get_parameter('search_pattern_radius').value
        self.search_pattern_speed = self.get_parameter('search_pattern_speed').value
        
        # State variables
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        
        # Tag tracking
        self.tag_detected = False
        self.tag_position = np.zeros(3)
        self.last_detection_time = None
        self.consecutive_detections = 0
        self.detections_needed = 5  # Need multiple consecutive detections to confirm tag
        
        # Current position estimate (start at search position)
        self.current_position = np.array([self.search_x, self.search_y, -self.search_altitude])
        
        # Landing states
        self.SEARCHING = 0       # Looking for tag
        self.CENTERING = 1       # Moving above tag
        self.DESCENDING = 2      # Descending to tag
        self.LANDED = 3          # Landed on tag
        self.landing_state = self.SEARCHING
        
        # Variables for linear approach
        self.dt = timer_period
        
        # Search pattern variables
        self.search_start_time = time.time()
        self.search_angle = 0.0
        
        self.get_logger().info('Linear precision landing node initialized')

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def get_tag_transform(self):
        """Get the transform to the AprilTag"""
        try:
            # Look up transform from camera to tag frame
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,  # Source frame (camera)
                self.tag_frame,     # Target frame (tag)
                rclpy.time.Time(),
                rclpy.time.Duration(seconds=0.1)  # Short timeout to avoid blocking
            )
            return transform
        except tf2_ros.LookupException:
            # Transform not available - tag not detected
            return None
        except tf2_ros.ExtrapolationException:
            # Transform available but too old
            return None
        except tf2_ros.ConnectivityException:
            # Cannot determine transform
            return None
        except Exception as e:
            self.get_logger().debug(f'Error getting transform: {str(e)}')
            return None

    def control_loop_callback(self):
        """Main control loop"""
        # Publish offboard control mode (required for offboard operation)
        self.publish_offboard_control_mode()
        
        # Only proceed if in offboard mode and armed
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and 
            self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            
            # Try to get the tag position
            tag_transform = self.get_tag_transform()
            
            if tag_transform is not None:
                # Got a transform reading
                current_time = time.time()
                
                # Update consecutive detections counter
                self.consecutive_detections += 1
                self.last_detection_time = current_time
                
                # Update tag position when detected
                self.tag_position = np.array([
                    tag_transform.transform.translation.x,
                    tag_transform.transform.translation.y,
                    tag_transform.transform.translation.z
                ])
                
                # Only consider tag detected after several consecutive detections
                if self.consecutive_detections >= self.detections_needed:
                    # Now we're confident it's a real detection
                    if not self.tag_detected:
                        self.get_logger().info('Tag confidently detected')
                    
                    self.tag_detected = True
                    
                    # If in SEARCHING state, transition to CENTERING
                    if self.landing_state == self.SEARCHING:
                        self.landing_state = self.CENTERING
                        self.get_logger().info('Transitioning to CENTERING')
            else:
                # No transform available
                current_time = time.time()
                
                # Reset consecutive detections counter
                self.consecutive_detections = 0
                
                # Check if detection timeout has elapsed
                if self.tag_detected and self.last_detection_time is not None:
                    if current_time - self.last_detection_time > self.detection_timeout:
                        # Tag lost for too long
                        self.tag_detected = False
                        self.get_logger().info('Tag lost - detection timeout')
                        
                        # If centering or descending, go back to searching
                        if self.landing_state in [self.CENTERING, self.DESCENDING]:
                            self.landing_state = self.SEARCHING
                            self.get_logger().info('Returning to SEARCHING')
            
            # Execute current state
            if self.landing_state == self.SEARCHING:
                self.execute_search()
            elif self.landing_state == self.CENTERING:
                self.execute_centering()
            elif self.landing_state == self.DESCENDING:
                self.execute_descent()
            elif self.landing_state == self.LANDED:
                self.execute_landed()

    def publish_offboard_control_mode(self):
        """Publish offboard control mode"""
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

    def execute_search(self):
        """Hover at search position or execute search pattern"""
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        if self.search_pattern_enabled:
            # Execute spiral search pattern
            current_time = time.time()
            elapsed_time = current_time - self.search_start_time
            
            # Update search angle
            self.search_angle = elapsed_time * self.search_pattern_speed
            
            # Calculate spiral pattern position
            radius = min(self.search_pattern_radius, 
                         0.1 + (elapsed_time * 0.2))  # Gradually increase radius
            
            x = self.search_x + radius * math.cos(self.search_angle)
            y = self.search_y + radius * math.sin(self.search_angle)
            
            trajectory_msg.position[0] = x
            trajectory_msg.position[1] = y
            trajectory_msg.position[2] = -self.search_altitude
            
            # Update current position estimate
            self.current_position = np.array([x, y, -self.search_altitude])
            
            # Log search pattern info occasionally
            if int(elapsed_time) % 5 == 0 and elapsed_time < 5.1:
                self.get_logger().info(f'Searching at: x={x:.2f}, y={y:.2f}, z={-self.search_altitude:.2f}')
        else:
            # Just hover at search position
            trajectory_msg.position[0] = self.search_x
            trajectory_msg.position[1] = self.search_y
            trajectory_msg.position[2] = -self.search_altitude
            
            # Update current position estimate
            self.current_position = np.array([
                self.search_x, 
                self.search_y, 
                -self.search_altitude
            ])
        
        self.publisher_trajectory.publish(trajectory_msg)

    def execute_centering(self):
        """Move to position directly above the tag"""
        if not self.tag_detected:
            return
        
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        # Calculate target position (above tag)
        target_position = np.array([
            self.tag_position[0],
            self.tag_position[1],
            -self.hover_altitude  # Hover above tag
        ])
        
        # Set position setpoint
        trajectory_msg.position[0] = target_position[0]
        trajectory_msg.position[1] = target_position[1]
        trajectory_msg.position[2] = target_position[2]
        
        # Update current position estimate
        self.current_position = target_position
        
        # Check if centered above tag within tolerance
        xy_error = np.sqrt((self.current_position[0] - self.tag_position[0])**2 + 
                           (self.current_position[1] - self.tag_position[1])**2)
        
        if xy_error < self.xy_tolerance:
            self.landing_state = self.DESCENDING
            self.get_logger().info(f'Centered above tag. Transitioning to DESCENDING')
        
        self.publisher_trajectory.publish(trajectory_msg)

    def execute_descent(self):
        """Descend onto the tag with continuous correction"""
        if not self.tag_detected:
            return
        
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        # Calculate target descent position
        # Keep x,y aligned with tag, but descend in z
        current_altitude = -self.current_position[2]
        new_altitude = max(0.0, current_altitude - self.descent_rate * self.dt)
        
        # Set position setpoint (always aligned with current tag position)
        trajectory_msg.position[0] = self.tag_position[0]
        trajectory_msg.position[1] = self.tag_position[1]
        trajectory_msg.position[2] = -new_altitude
        
        # Update current position estimate
        self.current_position = np.array([
            self.tag_position[0],
            self.tag_position[1],
            -new_altitude
        ])
        
        # Check if we've landed
        if new_altitude <= 0.05:  # 5cm from ground, consider landed
            self.landing_state = self.LANDED
            self.get_logger().info('Landing complete')
        
        self.publisher_trajectory.publish(trajectory_msg)

    def execute_landed(self):
        """Maintain position after landing"""
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        # Keep position at tag with zero altitude
        if self.tag_detected:
            trajectory_msg.position[0] = self.tag_position[0]
            trajectory_msg.position[1] = self.tag_position[1]
        else:
            # If tag lost after landing, stay at last known position
            trajectory_msg.position[0] = self.current_position[0]
            trajectory_msg.position[1] = self.current_position[1]
            
        trajectory_msg.position[2] = 0.0  # On ground
        
        self.publisher_trajectory.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    landing_node = LinearPrecisionLanding()
    rclpy.spin(landing_node)
    landing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
