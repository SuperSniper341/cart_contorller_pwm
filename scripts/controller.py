#!/usr/bin/env python3
"""
Cart Controller — Low-Level Control Layer
Sole purpose: Enforce strict hardware constraints (30ms response + STOP).
Exposes simple pulse_right/pulse_left/hold API for any navigator script.
"""

import math
import time
import numpy as np

try:
    import serial
except ImportError:
    serial = None

try:
    from carla_msgs.msg import CarlaEgoVehicleControl
except ImportError:
    CarlaEgoVehicleControl = None

from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

PULSE_MS           = 35.0     # Enforced minimum hardware pulse!
MAX_STEER_DEG      = 15.0
DEG_PER_PULSE      = 0.45
STEER_CMD_SPEED    = 15.0

MOTOR_RIGHT        = 51
MOTOR_LEFT         = 49
MOTOR_STOP         = 50

MAX_SPEED_KMH      = 7.0      
THROTTLE           = 0.4      
TARGET_THROTTLE    = 75       
WHEELBASE_M        = 1.8      
IMU_EMA_ALPHA      = 0.3      
SERIAL_PORT        = '/dev/ttyUSB0'
SERIAL_BAUD        = 9600

def build_serial_command(
    a_val=0, b_throttle=50, c_val=0, d_steering=50,
    e_left_indicator=0, f_horn=0, g_light=0,
    h_right_indicator=0, i_brake=0, j_reverse=0
):
    return (
        f"*A{a_val}B{b_throttle}C{c_val}D{d_steering}E{e_left_indicator}"
        f"F{f_horn}G{g_light}H{h_right_indicator}I{i_brake}J{j_reverse}#"
    )

class MotorSteering:
    """Discrete Hardware Abstraction Layer.
    Never modify this logic. It enforces a strict lifecycle:
    IDLE -> 51/49 for 35ms -> 50 (STOP) -> IDLE.
    """
    _IDLE     = 'IDLE'
    _PULSING  = 'PULSING'
    _STOPPING = 'STOPPING'

    def __init__(self):
        self.motor = MOTOR_STOP
        self._phase = self._IDLE
        self._pulse_start_ms = 0.0
        self.simulated_angle = 0.0

    def tick(self):
        if self._phase == self._PULSING:
            now_ms = time.time() * 1000.0
            if (now_ms - self._pulse_start_ms) >= PULSE_MS:
                self.motor = MOTOR_STOP
                self._phase = self._STOPPING
        elif self._phase == self._STOPPING:
            self._phase = self._IDLE
        return self.motor

    @property
    def is_idle(self):
        return self._phase == self._IDLE

    def hold(self):
        self.motor = MOTOR_STOP
        self._phase = self._IDLE

    def pulse_right(self):
        if self._phase != self._IDLE: return False
        self.motor = MOTOR_RIGHT
        self._phase = self._PULSING
        self._pulse_start_ms = time.time() * 1000.0
        self.simulated_angle = min(MAX_STEER_DEG, self.simulated_angle + DEG_PER_PULSE)
        return True

    def pulse_left(self):
        if self._phase != self._IDLE: return False
        self.motor = MOTOR_LEFT
        self._phase = self._PULSING
        self._pulse_start_ms = time.time() * 1000.0
        self.simulated_angle = max(-MAX_STEER_DEG, self.simulated_angle - DEG_PER_PULSE)
        return True

def quat_to_yaw_deg(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad) % 360.0

TOPICS = {
    'carla': { 'rgb': '/carla/hero/rgb/image', 'imu': '/carla/hero/imu', 'control': '/carla/hero/vehicle_control_cmd', 'steer': '/carla/hero/current_steer_angle' },
    'cart':  { 'rgb': '/bitsauto/rgb/image',   'imu': '/bitsauto/imu',   'control': None,                              'steer': '/bitsauto/current_steer_angle' },
}

class CartController:
    def __init__(self, node, mode='carla'):
        self.node = node
        self.mode = mode
        self.topics = TOPICS[mode]
        self.motor = MotorSteering()

        self.car_heading_deg = None
        self.smoothed_yaw_rate = 0.0
        self.actual_yaw_rate = 0.0
        self._imu_ok = False

        self.ser = None
        self.ctrl_pub = None
        self._last_serial_time = 0.0

        self._setup_imu()
        self._setup_control_output()

    def _setup_imu(self):
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.node.create_subscription(Imu, self.topics['imu'], self._on_imu, qos)

    def _setup_control_output(self):
        if self.mode == 'carla' and CarlaEgoVehicleControl and self.topics['control']:
            self.ctrl_pub = self.node.create_publisher(CarlaEgoVehicleControl, self.topics['control'], 10)
        elif self.mode == 'cart' and serial is not None:
            try:
                self.ser = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=0.05)
            except Exception as e:
                self.node.get_logger().warn(f'Serial failed: {e}')

    def _on_imu(self, msg):
        try:
            self.car_heading_deg = quat_to_yaw_deg(msg.orientation)
            raw_yaw = -math.degrees(msg.angular_velocity.z)
            self.actual_yaw_rate = raw_yaw
            if not self._imu_ok:
                self.smoothed_yaw_rate = raw_yaw
                self._imu_ok = True
            else:
                self.smoothed_yaw_rate = IMU_EMA_ALPHA * raw_yaw + (1 - IMU_EMA_ALPHA) * self.smoothed_yaw_rate
        except Exception:
            self._imu_ok = False

    @property
    def imu_ok(self): return self._imu_ok

    @property
    def imu_steer_deg(self):
        if not self._imu_ok: return 0.0
        speed_ms = MAX_SPEED_KMH / 3.6
        if speed_ms < 0.3: return 0.0
        return math.degrees(math.atan2(math.radians(self.smoothed_yaw_rate) * WHEELBASE_M, speed_ms))

    def tick(self):
        self.motor.tick()
        self._send_hw()

    def hold(self):
        self.motor.hold()
        self._send_hw()

    def _send_hw(self):
        if self.ctrl_pub and CarlaEgoVehicleControl:
            msg = CarlaEgoVehicleControl()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.steer = float(np.clip(self.motor.simulated_angle / MAX_STEER_DEG, -1.0, 1.0))
            msg.throttle = THROTTLE
            msg.brake = 0.0
            msg.hand_brake = False
            msg.reverse = False
            msg.gear = 1
            msg.manual_gear_shift = False
            self.ctrl_pub.publish(msg)

        if self.ser and self.ser.is_open:
            now = time.time()
            if (now - self._last_serial_time) >= 0.030:
                self._last_serial_time = now
                serial_cmd = build_serial_command(b_throttle=TARGET_THROTTLE, d_steering=self.motor.motor, i_brake=0)
                try:
                    self.ser.write(serial_cmd.encode())
                    while self.ser.readline(): pass
                except Exception as e:
                    pass

    def destroy(self):
        if self.ser and self.ser.is_open:
            self.ser.write(build_serial_command(b_throttle=50, d_steering=50, i_brake=1).encode())
            time.sleep(0.05)
            self.ser.close()
