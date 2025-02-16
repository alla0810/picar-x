#!/usr/bin/env python3
"""
Team IoT - CS437: Lab1B
Smart Navigation with Enhanced Mapping and Object Detection
"""

import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from heapq import heappush, heappop
from picamera2 import Picamera2
from picarx import Picarx

class EnhancedMapper:
    def __init__(self, px: Picarx):
        self.map_size = 150
        self.cell_size = 1.5
        self.grid = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.confidence = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.px = px
        self.position = np.array([self.map_size - 15, 15])
        self.heading = 135
        self.servo_offset = 0
        self.camera_tilt = 6
        self.safe_distance = 45
        self.danger_distance = 15
        self.scan_angles = range(-80, 81, 5)
        self.samples_per_angle = 5
        self.servo_settle_time = 0.3
        self.last_scan_time = time.time()
        self.rescan_interval = 5.0
        self.distance_threshold = 20
        self.last_scan_position = self.position.copy()
        self.total_movement = 0

    def should_rescan(self):
        current_time = time.time()
        time_since_scan = current_time - self.last_scan_time
        distance_moved = np.linalg.norm(self.position - self.last_scan_position)
        if time_since_scan >= self.rescan_interval:
            print("\nRescan triggered: Time interval reached")
            return True
        if distance_moved >= self.distance_threshold:
            print("\nRescan triggered: Significant movement detected")
            return True
        return False

    def read_distance(self, samples=5):
        readings = []
        for _ in range(samples):
            d = self.px.ultrasonic.read()
            if 0 <= d <= 200:
                readings.append(d)
            time.sleep(0.05)
        if not readings:
            return self.safe_distance
        readings = np.array(readings)
        median = np.median(readings)
        mad = np.median(np.abs(readings - median))
        valid = readings[abs(readings - median) <= 2.5 * mad]
        return np.mean(valid) if len(valid) > 0 else median

    def scan_environment(self, force_full_scan=False):
        if not force_full_scan and not self.should_rescan():
            return None
        print("\nPerforming enhanced environment scan...")
        self.confidence *= 0.8
        scan_results = []
        for angle in range(-80, 81, 5):
            self.px.set_cam_pan_angle(angle)
            time.sleep(self.servo_settle_time)
            distance = self.read_distance(5)
            scan_results.append((angle, distance))
            self._update_grid_with_shape(angle, distance)
            print(f"Detailed scan angle {angle:3d}°: {distance:4.1f}cm")
        self.px.set_cam_pan_angle(0)
        self.last_scan_time = time.time()
        self.last_scan_position = self.position.copy()
        self._enhance_shapes()
        self._update_map_visualization()
        return scan_results

    def _update_grid_with_shape(self, sensor_angle, distance):
        """
        Update grid cells based on sensor reading, with improved obstacle marking
        """
        total_angle = (self.heading + sensor_angle) % 360
        rad_angle = math.radians(total_angle)
        x, y = self.position
        end_x = x + math.cos(rad_angle) * (distance / self.cell_size)
        end_y = y + math.sin(rad_angle) * (distance / self.cell_size)
        steps = int(distance / self.cell_size)
        if steps <= 0:
            return
        dx = (end_x - x) / steps
        dy = (end_y - y) / steps
        for i in range(steps):
            cell_x = int(x + dx * i)
            cell_y = int(y + dy * i)
            if 0 <= cell_x < self.map_size and 0 <= cell_y < self.map_size:
                self.grid[cell_y, cell_x] = 0
                self.confidence[cell_y, cell_x] = 1.0
        if distance < self.safe_distance:
            obstacle_x = int(end_x)
            obstacle_y = int(end_y)
            if 0 <= obstacle_x < self.map_size and 0 <= obstacle_y < self.map_size:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx = obstacle_x + dx
                        ny = obstacle_y + dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            self.grid[ny, nx] = 1
                            self.confidence[ny, nx] = 1.0

    def _mark_obstacle_shape(self, x, y, sensor_angle):
        """
        Create precise line drawings of detected obstacles,
        optimized for detecting and drawing box-like structures.
        """
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return
        if not hasattr(self, 'scan_history'):
            self.scan_history = []
            self.last_scan_distances = {}
            self.edge_points = set()
        current_distance = math.sqrt((x - self.position[0])**2 + (y - self.position[1])**2)
        rad_angle = math.radians(self.heading + sensor_angle)
        exact_x = self.position[0] + current_distance * math.cos(rad_angle)
        exact_y = self.position[1] + current_distance * math.sin(rad_angle)
        store_point = False
        if sensor_angle in self.last_scan_distances:
            prev_dist = self.last_scan_distances[sensor_angle]
            if abs(current_distance - prev_dist) > 3:
                store_point = True
        else:
            store_point = True
        if store_point:
            self.scan_history.append((exact_x, exact_y, sensor_angle))
            self.edge_points.add((int(exact_x), int(exact_y)))
        self.last_scan_distances[sensor_angle] = current_distance
        max_history = 25
        if len(self.scan_history) > max_history:
            self.scan_history = self.scan_history[-max_history:]
        if len(self.scan_history) > 1:
            prev_x, prev_y, prev_angle = self.scan_history[-2]
            angle_diff = abs(prev_angle - sensor_angle)
            point_dist = math.sqrt((prev_x - exact_x)**2 + (prev_y - exact_y)**2)
            if angle_diff < 15 and point_dist < 10:
                self._draw_line(int(prev_x), int(prev_y), int(exact_x), int(exact_y))

    def _draw_line(self, x1, y1, x2, y2):
        """
        Draw a single-pixel width line between two points.
        Uses Bresenham's algorithm for smooth, continuous lines.
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.grid[y, x] = 1.0
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _enhance_shapes(self):
        """Enhance detected shapes while preserving obstacle information"""
        from scipy.ndimage import gaussian_filter
        obstacles = self.grid > 0.5
        smoothed = gaussian_filter(self.grid, sigma=0.8)
        self.grid = np.where(obstacles, 1.0, smoothed)

    def _update_map_visualization(self):
        """
        Display two different visualizations:
        1. A colored pixel map showing the environment
        2. A binary grid showing obstacles as 1's and clear space as 0's
        """
        occupied = np.where(self.grid > 0.2)
        if len(occupied[0]) == 0:
            print("Map is empty - no obstacles detected")
            return
        GREEN_BG = "\033[42m"
        RED_FG = "\033[91m"
        BLUE_FG = "\033[94m"
        RESET = "\033[0m"
        PIXEL = "  "
        LINE = "██"
        y_min, y_max = np.min(occupied[0]), np.max(occupied[0])
        x_min, x_max = np.min(occupied[1]), np.max(occupied[1])
        margin = 12
        y_min = max(0, y_min - margin)
        y_max = min(self.map_size - 1, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(self.map_size - 1, x_max + margin)
        print("\nEnvironment Map:")
        for y in range(y_min, y_max + 1):
            line_str = ""
            for x in range(x_min, x_max + 1):
                if int(self.position[0]) == x and int(self.position[1]) == y:
                    line_str += f"{BLUE_FG}{LINE}{RESET}"
                elif self.grid[y, x] > 0.7:
                    line_str += f"{RED_FG}{LINE}{RESET}"
                else:
                    line_str += f"{GREEN_BG}{PIXEL}{RESET}"
            print(line_str)
        print("\nBinary Occupancy Grid (1=obstacle, 0=clear):")
        print("-" * (x_max - x_min + 1))
        for y in range(y_min, y_max + 1):
            binary_line = ""
            for x in range(x_min, x_max + 1):
                if self.grid[y, x] > 0.5:
                    binary_line += "1"
                else:
                    binary_line += "0"
            print(binary_line)
        print("-" * (x_max - x_min + 1))

    def update_position(self, movement_vector, angle_change=0):
        old_pos = self.position.copy()
        self.position += movement_vector
        self.heading = (self.heading + angle_change) % 360
        movement = np.linalg.norm(self.position - old_pos)
        self.total_movement += movement
        self.position[0] = np.clip(self.position[0], 0, self.map_size - 1)
        self.position[1] = np.clip(self.position[1], 0, self.map_size - 1)

def classify_traffic_light(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
    red_mask2 = cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, np.array([40,120,70]), np.array([80,255,255]))
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)
    if red_pixels > green_pixels and red_pixels > 100:
        return "Red Light", (0, 0, 255)
    elif green_pixels > red_pixels and green_pixels > 100:
        return "Green Light", (0, 255, 0)
    else:
        return "Traffic Light", (255, 255, 0)

def visualize_detection(frame, detection_result):
    MARGIN = 10
    ROW_SIZE = 30
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (0, 0, 0)
    for detection in detection_result.detections:
        cat = detection.categories[0]
        label = cat.category_name
        score = cat.score
        if label not in ("stop sign", "traffic light"):
            continue
        if score < 0.2:
            continue
        bbox = detection.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        if label == "traffic light":
            sublabel, box_color = classify_traffic_light(roi)
            final_label = f"{sublabel} ({score:.2f})"
        else:
            box_color = (0, 0, 255)
            final_label = f"Stop Sign ({score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, final_label, (x1 + MARGIN, y1 + ROW_SIZE),
                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return frame

class TeamIoT_SmartNavigator:
    def __init__(self):
        self.px = Picarx()
        self.servo_offset = -6.1
        self.forward_speed = 22
        self.slow_speed = 15
        self.backup_speed = 8
        self.safe_distance = 43
        self.danger_distance = 19
        self.very_danger_distance = 15
        self.mapper = EnhancedMapper(self.px)
        self.px.set_cam_tilt_angle(8)
        time.sleep(0.2)
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (640, 480)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        base_opts = python.BaseOptions(model_asset_path="efficientdet_lite0.tflite")
        detect_opts = vision.ObjectDetectorOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=5,
            score_threshold=0.35,
            result_callback=self._process_detections
        )
        self.detector = vision.ObjectDetector.create_from_options(detect_opts)
        self.detection_result_list = []
        self.current_detections = []
        self.counter = 0
        self.fps = 0
        self.start_time = time.time()
        self.fps_avg_frame_count = 10
        self.stop_sign_detected = False
        self.red_light_detected = False
        self.green_light_detected = False
        self.last_stop_time = 0
        self.STOP_WAIT = 3.0
        self.map_size = 100
        self.grid = np.zeros((self.map_size, self.map_size), dtype=int)
        self.CELL_SIZE_CM = 2
        self.position = (self.map_size - 10, 10)
        self.heading = 135
        self.WIDE_SCAN_MIN = -80
        self.WIDE_SCAN_MAX = 80
        self.WIDE_STEP = 10
        self.set_steering(0)
        self.px.set_cam_pan_angle(0)
        time.sleep(0.2)

    def _process_detections(self, detection_result, image, timestamp_ms):
        if self.counter % self.fps_avg_frame_count == 0:
            self.fps = self.fps_avg_frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
        self.detection_result_list.append(detection_result)
        self.counter += 1
        self.current_detections = []
        self.stop_sign_detected = False
        self.red_light_detected = False
        self.green_light_detected = False
        for detection in detection_result.detections:
            cat = detection.categories[0]
            label = cat.category_name
            score = cat.score
            if score < 0.35:
                continue
            if label == "stop sign":
                self.stop_sign_detected = True
                self.current_detections.append(detection)
            elif label == "traffic light":
                self.red_light_detected = True
                self.current_detections.append(detection)

    def update_camera_and_detect(self):
        frame = self.picam2.capture_array()
        cv2.putText(frame, f"FPS={self.fps:.1f}", (24,50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.detector.detect_async(mp_image, time.time_ns()//1_000_000)
        if self.detection_result_list:
            detection_result = self.detection_result_list.pop(0)
            frame = visualize_detection(frame, detection_result)
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(1)

    def check_traffic_logic(self):
        now = time.time()
        if self.stop_sign_detected and (now - self.last_stop_time > self.STOP_WAIT*2):
            print("Stop sign => stop 3s.")
            self.stop()
            time.sleep(self.STOP_WAIT)
            self.last_stop_time = now
        elif self.red_light_detected and (now - self.last_stop_time > self.STOP_WAIT):
            print("Traffic light => red => stop 3s.")
            self.stop()
            time.sleep(self.STOP_WAIT)
            self.last_stop_time = now

    def set_steering(self, angle):
        angle = max(-33, min(33, angle))
        self.px.set_dir_servo_angle(angle + self.servo_offset)
        time.sleep(0.2)

    def forward(self, speed=None, duration=None):
        if speed is None:
            speed = self.forward_speed
        self.px.forward(speed)
        if duration:
            time.sleep(duration)
            self.stop()

    def backward(self, speed=None, duration=0.7):
        if speed is None:
            speed = self.backup_speed
        self.px.backward(speed)
        time.sleep(duration)
        self.stop()

    def stop(self):
        self.px.forward(0)

    def read_ultrasonic(self, samples=5):
        total = 0
        valid = 0
        for _ in range(samples):
            d = self.px.ultrasonic.read()
            if 0 <= d <= 200:
                total += d
                valid += 1
            time.sleep(0.03)
        return total / valid if valid > 0 else 100

    def scan_angle(self, angle, settle=0.2):
        self.px.set_cam_pan_angle(angle)
        time.sleep(settle)
        return self.read_ultrasonic(3)

    def wide_scan(self):
        print("\n--- Starting enhanced wide scan ---")
        scan_results = self.mapper.scan_environment(force_full_scan=True)
        self.grid = self.mapper.grid.copy()
        self._print_map()

    def _print_map(self):
        """Print the occupancy grid with improved visibility"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        occupied = np.where(self.grid > 0.5)
        if len(occupied[0]) == 0:
            print("Map is empty - no obstacles.")
            return
        y_min, y_max = np.min(occupied[0]), np.max(occupied[0])
        x_min, x_max = np.min(occupied[1]), np.max(occupied[1])
        margin = 5
        min_size = 20
        y_min = max(0, y_min - margin)
        y_max = min(self.map_size - 1, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(self.map_size - 1, x_max + margin)
        print("\nOccupancy Grid:")
        print("-" * (x_max - x_min + 2*margin + 1))
        for y in range(y_min, y_max + 1):
            line = ""
            for x in range(x_min, x_max + 1):
                if self.grid[y, x] > 0.8:
                    line += f"{RED}1{RESET}"
                else:
                    line += f"{GREEN}0{RESET}"
            print(line)
        print("-" * (x_max - x_min + 2*margin + 1))

    def find_path(self, gx, gy):
        start = self.position
        goal = (gx, gy)
        if self.grid[goal[1], goal[0]] == 1:
            print("Goal is blocked!")
            return None
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        moves = [(1,0), (-1,0), (0,1), (0,-1),
                 (1,1), (1,-1), (-1,1), (-1,-1)]
        while frontier:
            _, current = heappop(frontier)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            for dx, dy in moves:
                nx = current[0] + dx
                ny = current[1] + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.grid[ny, nx] == 1:
                        continue
                    new_cost = cost_so_far[current] + 1
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        priority = new_cost + heuristic((nx, ny), goal)
                        heappush(frontier, (priority, (nx, ny)))
                        came_from[(nx, ny)] = current
        return None

    def scan3(self):
        left_d  = self.scan_angle(-80)
        center_d = self.scan_angle(0)
        right_d = self.scan_angle(80)
        self.px.set_cam_pan_angle(0)
        time.sleep(0.1)
        return {"left": left_d, "center": center_d, "right": right_d}

    def pick_best_direction(self, dists):
        good = {k: v for (k, v) in dists.items() if v >= self.safe_distance}
        if not good:
            return "none"
        if "center" in good:
            return "center"
        ld = good.get("left", -1)
        rd = good.get("right", -1)
        return "left" if ld >= rd else "right"

    def pivot_turn(self, direction, duration=1.2, forward=True):
        """
        Implements a sharp pivot turn, similar to a tank-style turn.
        Instead of making a wide arc, the car will turn more aggressively in place,
        which helps avoid hitting obstacles with the bumpers during turns.
        """
        if direction == "left":
            # Sharper angle for left turn, close to maximum
            angle = -40  # Using a more aggressive angle
            angle_change = -45
            print("Pivot turn LEFT")
            
            # Brief back-and-turn motion to initiate the pivot
            self.set_steering(angle)
            self.backward(speed=self.backup_speed, duration=0.3)
            self.forward(speed=self.slow_speed, duration=1.0)
            
        else:
            # Mirror settings for right turn
            angle = 30
            angle_change = 45
            print("Pivot turn RIGHT")
            
            # Brief back-and-turn motion to initiate the pivot
            self.set_steering(angle)
            self.backward(speed=self.backup_speed, duration=0.3)
            self.forward(speed=self.slow_speed, duration=0.8)

        # Straighten out and finish the turn
        self.set_steering(0)
        self.forward(speed=self.slow_speed, duration=0.4)
        
        self.mapper.heading = (self.mapper.heading + angle_change) % 360
        print("Turn complete => rescanning map.")
        self.wide_scan()

    def move_to_cell(self, x, y):
        print(f"\nMove to cell ({x},{y})")
        while True:
            self.update_camera_and_detect()
            self.check_traffic_logic()
            dist = self.read_ultrasonic(samples=3)
            if dist >= self.safe_distance:
                print("Safe => forward 0.5s")
                self.set_steering(0)
                self.forward(duration=0.5)
                self.position = (x, y)
                movement_vector = np.array([x, y]) - self.mapper.position
                self.mapper.update_position(movement_vector)
                return True
            elif dist < self.danger_distance:
                print("Too close => backing up => re-scan for new map.")
                self.backward(duration=1.0)
                self.wide_scan()
                return False
            else:
                print("Obstacle => scanning L/C/R.")
                triple = self.scan3()
                best = self.pick_best_direction(triple)
                if best == "none":
                    print("No path => back up => re-scan.")
                    self.backward(duration=1.0)
                    self.wide_scan()
                    return False
                elif best == "center":
                    print("Center => slow forward 0.5s")
                    self.forward(speed=self.slow_speed, duration=0.5)
                else:
                    self.pivot_turn(best, duration=1.5, forward=True)
                    return False
            time.sleep(0.2)

    def navigate_to_goal(self, gx, gy):
        print(f"\nPlanning path to goal=({gx},{gy})")
        self.wide_scan()
        while True:
            path = self.find_path(gx, gy)
            if not path:
                print("No valid path found!")
                return False
            print(f"Found path with {len(path)-1} steps.")
            cells = path[1:]
            for cell in cells:
                success = self.move_to_cell(*cell)
                if not success:
                    break
                if cell == (gx, gy):
                    print("Reached final goal!")
                    self.stop()
                    return True

    def run(self):
        GOAL_X = 83
        GOAL_Y = 14
        self.navigate_to_goal(GOAL_X, GOAL_Y)
        print("Navigation complete.")

def main():
    nav = TeamIoT_SmartNavigator()
    try:
        nav.run()
    except KeyboardInterrupt:
        nav.stop()
        print("\nUser interrupted.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
