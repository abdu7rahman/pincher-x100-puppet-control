#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from interbotix_xs_msgs.msg import JointGroupCommand
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import cv2
import mediapipe as mp
import numpy as np
from threading import Thread, Lock

class PincherFullControl(Node):
    def __init__(self):
        super().__init__('pincher_full_control')
        
        self.arm_pub = self.create_publisher(JointGroupCommand, '/px100/commands/joint_group', 10)
        
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.get_logger().info('Waiting for MoveIt...')
        self._execute_client.wait_for_server()
        self.get_logger().info('Connected!')
        
        self.current_arm_joints = None
        self.current_gripper_position = None
        self.joint_sub = self.create_subscription(JointState, '/px100/joint_states', self.joint_callback, 10)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.joint_limits = [(-3.14, 3.14), (-1.8, 1.5), (-1.8, 1.5), (-1.8, 1.8)]
        self.smoothing = 0.7
        self.prev_joints = None
        self.wrist_angle = 1.23
        
        self.gripper_released = 0.037
        self.gripper_grasping = -0.037
        self.pinch_threshold = 0.06
        self.current_gripper_state = "Released"
        
        self.pinch_history = []
        self.history_size = 5
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        self.hand_detected = False
        self.gripper_executing = False
        self.lock = Lock()
        
        print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))
        
    def joint_callback(self, msg):
        try:
            self.current_arm_joints = list(msg.position[:4])
            if 'left_finger' in msg.name:
                idx = msg.name.index('left_finger')
                self.current_gripper_position = msg.position[idx]
        except:
            pass
    
    def calculate_pinch_distance(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        return distance
    
    def smooth_pinch_detection(self, current_distance):
        self.pinch_history.append(current_distance)
        if len(self.pinch_history) > self.history_size:
            self.pinch_history.pop(0)
        return np.mean(self.pinch_history)
    
    def control_gripper(self, is_pinched):
        target_state = "Grasping" if is_pinched else "Released"
        
        if target_state != self.current_gripper_state:
            with self.lock:
                if self.gripper_executing:
                    return
                self.gripper_executing = True
            
            try:
                if self.current_gripper_position is None:
                    with self.lock:
                        self.gripper_executing = False
                    return
                
                target_position = self.gripper_grasping if is_pinched else self.gripper_released
                
                trajectory = JointTrajectory()
                trajectory.joint_names = ['left_finger', 'right_finger']
                
                point1 = JointTrajectoryPoint()
                point1.positions = [self.current_gripper_position, self.current_gripper_position]
                point1.time_from_start.sec = 0
                point1.time_from_start.nanosec = 0
                
                point2 = JointTrajectoryPoint()
                point2.positions = [target_position, target_position]
                point2.time_from_start.sec = 0
                point2.time_from_start.nanosec = 200_000_000
                
                trajectory.points.append(point1)
                trajectory.points.append(point2)
                
                robot_trajectory = RobotTrajectory()
                robot_trajectory.joint_trajectory = trajectory
                
                goal_msg = ExecuteTrajectory.Goal()
                goal_msg.trajectory = robot_trajectory
                
                self._execute_client.send_goal_async(goal_msg)
                self.current_gripper_state = target_state
                
            except Exception as e:
                self.get_logger().error(f'Gripper failed: {e}')
            finally:
                with self.lock:
                    self.gripper_executing = False
    
    def hand_to_joint_angles(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        
        hand_x = index_tip.x
        hand_y = index_tip.y
        hand_z = index_tip.z
        
        joint1 = np.interp(hand_x, [0, 1], [self.joint_limits[0][1], self.joint_limits[0][0]])
        joint2 = np.interp(hand_y, [0, 1], [self.joint_limits[1][0], self.joint_limits[1][1]])
        joint3 = np.interp(hand_z, [-0.15, 0.15], [self.joint_limits[2][0], self.joint_limits[2][1]])
        joint4 = self.wrist_angle
        
        return [joint1, joint2, joint3, joint4]
    
    def smooth_joints(self, new_joints):
        if self.prev_joints is None:
            self.prev_joints = new_joints
            return new_joints
        
        smoothed = [
            self.smoothing * prev + (1 - self.smoothing) * new
            for new, prev in zip(new_joints, self.prev_joints)
        ]
        self.prev_joints = smoothed
        return smoothed
    
    def move_arm(self, joint_angles):
        try:
            msg = JointGroupCommand()
            msg.name = 'arm'
            msg.cmd = joint_angles
            self.arm_pub.publish(msg)
            return True
        except Exception as e:
            self.get_logger().error(f'Arm move failed: {e}')
            return False
    
    def control_loop(self):
        while self.is_running and rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                self.hand_detected = True
                
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    target_joints = self.hand_to_joint_angles(hand_landmarks)
                    target_joints = self.smooth_joints(target_joints)
                    self.move_arm(target_joints)
                    
                    raw_distance = self.calculate_pinch_distance(hand_landmarks)
                    smoothed_distance = self.smooth_pinch_detection(raw_distance)
                    is_pinched = smoothed_distance < self.pinch_threshold
                    
                    if not self.gripper_executing:
                        self.control_gripper(is_pinched)
                    
                    h, w = frame.shape[:2]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    line_color = (0, 0, 255) if is_pinched else (255, 255, 0)
                    line_thickness = 4 if is_pinched else 2
                    cv2.line(frame, (tx, ty), (ix, iy), line_color, line_thickness)
                    cv2.circle(frame, (tx, ty), 8, line_color, -1)
                    
                    cx, cy = ix, iy
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
                    cv2.line(frame, (cx-15, cy), (cx+15, cy), (0, 0, 255), 2)
                    cv2.line(frame, (cx, cy-15), (cx, cy+15), (0, 0, 255), 2)
                    
                    gripper_color = (0, 0, 255) if is_pinched else (0, 255, 0)
                    cv2.putText(frame, self.current_gripper_state.upper(), 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gripper_color, 2)
            else:
                self.hand_detected = False
            
            status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
            cv2.circle(frame, (620, 20), 15, status_color, -1)
            
            frame = cv2.resize(frame, (1280, 960))
            cv2.imshow('PincherX100', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('r'):
                self.reset_robot()
            elif key == ord('+') or key == ord('='):
                self.smoothing = max(0.1, self.smoothing - 0.1)
            elif key == ord('-') or key == ord('_'):
                self.smoothing = min(0.9, self.smoothing + 0.1)
            elif key == ord('t'):
                self.gripper_grasping = max(-0.05, self.gripper_grasping - 0.005)
            elif key == ord('g'):
                self.gripper_grasping = min(-0.01, self.gripper_grasping + 0.005)
    
    def reset_robot(self):
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = [0.0, -1.23, 0.0, 1.23]
        self.arm_pub.publish(msg)
        self.prev_joints = None
        self.current_gripper_state = "Released"
    
    def cleanup(self):
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    controller = PincherFullControl()
    
    control_thread = Thread(target=controller.control_loop, daemon=True)
    control_thread.start()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
