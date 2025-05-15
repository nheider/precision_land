#!/usr/bin/env python3

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
from std_msgs.msg import String  # For camera commands
from std_msgs.msg import Float32MultiArray  # For tag position
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue  # For node status

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
        
        # Publisher for camera control commands
        self.camera_cmd_pub = self.create_publisher(
            String,
            '/siyi_a8/command',
            10)
        
        # Status publisher
        self.status_pub = self.create_publisher(
            DiagnosticArray,
            '/linear_precision_landing/status',
            10)
        
        # Tag position publisher (for visualization/debugging)
        self.tag_pos_pub = self.create_publisher(
            Float32MultiArray,
            '/linear_precision_landing/tag_position',
            10)
        
        # TF listener for AprilTag pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer for control loop
        self.dt = 0.02  # seconds
        self.timer = self.create_timer(self.dt, self.control_loop_callback)
        
        # Timer for status publishing (1Hz)
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        # Parameters
        self.declare_parameter('tag_frame', 'marker')
        self.declare_parameter('camera_frame', 'gimbal_camera_link')
        self.declare_parameter('search_altitude', 2.0)
        self.declare_parameter('hover_altitude', 1.0)
        self.declare_parameter('descent_rate', 0.1)  # m/s
        self.declare_parameter('search_x', 0.0)
        self.declare_parameter('search_y', 0.0)
        self.declare_parameter('xy_tolerance', 0.1)  # 10cm tolerance
        self.declare_parameter('detection_timeout', 1.0)
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
        
        # Tag detection
        self.tag_detected = False
        self.tag_position = np.zeros(3)
        self.last_detection_time = None
        self.consecutive_detections = 0
        self.detections_needed = 5
        
        # Landing states
        self.SEARCHING = 0
        self.CENTERING = 1
        self.DESCENDING = 2
        self.LANDED = 3
        self.landing_state = self.SEARCHING
        self.landing_state_names = {
            self.SEARCHING: "SEARCHING",
            self.CENTERING: "CENTERING",
            self.DESCENDING: "DESCENDING",
            self.LANDED: "LANDED"
        }
        
        # Current position tracking
        self.current_position = np.array([self.search_x, self.search_y, -self.search_altitude])
        
        # Search pattern
        self.search_start_time = time.time()
        self.search_angle = 0.0
        
        # Status tracking
        self.detection_count = 0
        self.status_message = "Initialized"
        self.last_state_change_time = time.time()
        
        self.get_logger().info('Precision landing node initialized')

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def get_tag_transform(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.tag_frame,
                rclpy.time.Time(),
                rclpy.time.Duration(seconds=0.1)
            )
        except Exception:
            return None

    def control_loop_callback(self):
        # Publish offboard control mode
        self.publish_offboard_control_mode()

        # Always command camera down when offboard engaged & armed
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            cmd = String()
            cmd.data = 'straight_down'
            self.camera_cmd_pub.publish(cmd)

        # Only proceed with landing when offboard engaged & armed
        if not (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            return

        # Record previous state for change detection
        prev_state = self.landing_state

        # AprilTag detection
        transform = self.get_tag_transform()
        if transform:
            self.consecutive_detections += 1
            self.last_detection_time = time.time()
            self.tag_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Publish tag position for visualization/debugging
            tag_pos_msg = Float32MultiArray()
            tag_pos_msg.data = self.tag_position.tolist()
            self.tag_pos_pub.publish(tag_pos_msg)
            
            if self.consecutive_detections >= self.detections_needed:
                if not self.tag_detected:
                    self.get_logger().info('Tag confidently detected')
                    self.status_message = "Tag detected"
                self.tag_detected = True
                self.detection_count += 1
                if self.landing_state == self.SEARCHING:
                    self.landing_state = self.CENTERING
                    self.status_message = "Transitioning to CENTERING"
                    self.get_logger().info('Transitioning to CENTERING')
        else:
            if self.tag_detected and self.last_detection_time and \
               (time.time() - self.last_detection_time > self.detection_timeout):
                self.tag_detected = False
                self.status_message = "Tag lost"
                if self.landing_state in [self.CENTERING, self.DESCENDING]:
                    self.landing_state = self.SEARCHING
                    self.status_message = "Returning to SEARCHING - tag lost"
                    self.get_logger().info('Returning to SEARCHING')
            self.consecutive_detections = 0

        # Track state changes
        if prev_state != self.landing_state:
            self.last_state_change_time = time.time()

        # Execute state actions
        if self.landing_state == self.SEARCHING:
            self.execute_search()
        elif self.landing_state == self.CENTERING:
            self.execute_centering()
        elif self.landing_state == self.DESCENDING:
            self.execute_descent()
        elif self.landing_state == self.LANDED:
            self.execute_landed()

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.publisher_offboard_mode.publish(msg)

    def execute_search(self):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        if self.search_pattern_enabled:
            elapsed = time.time() - self.search_start_time
            self.search_angle = elapsed * self.search_pattern_speed
            radius = min(self.search_pattern_radius, 0.1 + elapsed * 0.2)
            x = self.search_x + radius * math.cos(self.search_angle)
            y = self.search_y + radius * math.sin(self.search_angle)
            msg.position[0], msg.position[1], msg.position[2] = x, y, -self.search_altitude
            self.current_position = np.array([x, y, -self.search_altitude])
        else:
            msg.position[0], msg.position[1], msg.position[2] = \
                self.search_x, self.search_y, -self.search_altitude
            self.current_position = np.array([self.search_x, self.search_y, -self.search_altitude])
        self.publisher_trajectory.publish(msg)

    def execute_centering(self):
        if not self.tag_detected:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        tx, ty = self.tag_position[:2]
        msg.position[0], msg.position[1], msg.position[2] = \
            tx, ty, -self.hover_altitude
        self.current_position = np.array([tx, ty, -self.hover_altitude])
        error = np.linalg.norm(self.current_position[:2] - self.tag_position[:2])
        if error < self.xy_tolerance:
            self.landing_state = self.DESCENDING
            self.status_message = "Centered above tag, descending"
            self.get_logger().info('Centered above tag, descending')
        self.publisher_trajectory.publish(msg)

    def execute_descent(self):
        if not self.tag_detected:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        current_alt = -self.current_position[2]
        new_alt = max(0.0, current_alt - self.descent_rate * self.dt)
        tx, ty = self.tag_position[:2]
        msg.position[0], msg.position[1], msg.position[2] = tx, ty, -new_alt
        self.current_position = np.array([tx, ty, -new_alt])
        if new_alt <= 0.05:
            self.landing_state = self.LANDED
            self.status_message = "Landing complete"
            self.get_logger().info('Landing complete')
        self.publisher_trajectory.publish(msg)

    def execute_landed(self):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        if self.tag_detected:
            tx, ty = self.tag_position[:2]
        else:
            tx, ty = self.current_position[:2]
        msg.position[0], msg.position[1], msg.position[2] = tx, ty, 0.0
        self.publisher_trajectory.publish(msg)

    def publish_status(self):
        """Publish detailed node status at regular intervals"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        # Create status message
        status = DiagnosticStatus()
        status.name = "linear_precision_landing"
        
        # Set overall status level
        if not (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            status.level = DiagnosticStatus.WARN
            status.message = "Not in OFFBOARD mode"
        elif not self.tag_detected and self.landing_state != self.SEARCHING:
            status.level = DiagnosticStatus.WARN
            status.message = "Tag not detected"
        else:
            status.level = DiagnosticStatus.OK
            status.message = self.status_message
        
        # Add detailed values
        status.values = [
            KeyValue(key="landing_state", value=self.landing_state_names[self.landing_state]),
            KeyValue(key="nav_state", value=str(self.nav_state)),
            KeyValue(key="arming_state", value=str(self.arming_state)),
            KeyValue(key="tag_detected", value=str(self.tag_detected)),
            KeyValue(key="consecutive_detections", value=str(self.consecutive_detections)),
            KeyValue(key="total_detections", value=str(self.detection_count)),
            KeyValue(key="current_position_x", value=f"{self.current_position[0]:.2f}"),
            KeyValue(key="current_position_y", value=f"{self.current_position[1]:.2f}"),
            KeyValue(key="current_position_z", value=f"{self.current_position[2]:.2f}"),
            KeyValue(key="current_altitude", value=f"{-self.current_position[2]:.2f}"),
            KeyValue(key="time_in_current_state", value=f"{time.time() - self.last_state_change_time:.1f}"),
        ]
        
        # Add tag position if detected
        if self.tag_detected:
            status.values.extend([
                KeyValue(key="tag_position_x", value=f"{self.tag_position[0]:.2f}"),
                KeyValue(key="tag_position_y", value=f"{self.tag_position[1]:.2f}"),
                KeyValue(key="tag_position_z", value=f"{self.tag_position[2]:.2f}"),
                KeyValue(key="xy_error", value=f"{np.linalg.norm(self.current_position[:2] - self.tag_position[:2]):.3f}"),
            ])
        
        diag_array.status = [status]
        self.status_pub.publish(diag_array)


def main(args=None):
    rclpy.init(args=args)
    node = LinearPrecisionLanding()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
