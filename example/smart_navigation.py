#!/usr/bin/env python3
"""
Team IoT - CS437: Lab1B
Full code that:
  - Rescans/updates the map whenever we turn or back up (periodic rescan).
  - Only detects "stop sign" or "traffic light" from the COCO dataset.
  - Classifies traffic light color as red or green in HSV.
  - Uses A* pathfinding and moves cell-by-cell.
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


def classify_traffic_light(bgr_roi):
    """
    Analyze bounding box (ROI) for red vs. green color.
    Return (label, box_color), e.g. ("Red Light",(0,0,255)) or ("Green Light",(0,255,0)).
    """
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # Red mask (two segments)
    red_mask1 = cv2.inRange(hsv, np.array([0,   120, 70]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180,255,255]))
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    # Green mask
    green_mask= cv2.inRange(hsv, np.array([40,120,70]), np.array([80,255,255]))

    red_pixels   = np.sum(red_mask>0)
    green_pixels = np.sum(green_mask>0)

    # Decide which color is dominant
    if red_pixels > green_pixels and red_pixels > 100:
        return "Red Light", (0, 0, 255)
    elif green_pixels > red_pixels and green_pixels > 100:
        return "Green Light", (0, 255, 0)
    else:
        # Fallback "Traffic Light" (yellow box)
        return "Traffic Light", (255, 255, 0)


def visualize_detection(frame, detection_result):
    """
    Draw bounding boxes for "stop sign" or "traffic light".
    If traffic light, do color analysis (red vs. green).
    """
    MARGIN = 10
    ROW_SIZE = 30
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (0, 0, 0)

    for detection in detection_result.detections:
        cat = detection.categories[0]
        label = cat.category_name
        score = cat.score

        # Only keep "stop sign" or "traffic light"
        if label not in ("stop sign", "traffic light"):
            continue
        if score < 0.2:
            continue

        bbox = detection.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height

        # Clamp coords
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
            # stop sign => red box
            box_color = (0, 0, 255)
            final_label = f"Stop Sign ({score:.2f})"

        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 3)
        cv2.putText(frame, final_label, (x1 + MARGIN, y1 + ROW_SIZE),
                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return frame


class TeamIoT_SmartNavigator:
    def __init__(self):
        """
        Initialize the car, camera, TFLite detection, mapping, etc.
        """
        self.px = Picarx()
        self.servo_offset = -6.1
        self.forward_speed = 22
        self.slow_speed = 15
        self.backup_speed = 8
        self.safe_distance = 43
        self.danger_distance = 19
        self.very_danger_distance = 15

        # Slight camera tilt
        self.px.set_cam_tilt_angle(8)
        time.sleep(0.2)

        # Camera + TFLite
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

        # For storing detection results
        self.detection_result_list = []
        self.current_detections = []
        self.counter = 0
        self.fps = 0
        self.start_time = time.time()
        self.fps_avg_frame_count = 10

        # Stop sign / traffic light flags
        self.stop_sign_detected = False
        self.red_light_detected = False
        self.green_light_detected = False
        self.last_stop_time = 0
        self.STOP_WAIT = 3.0

        # Occupancy grid for mapping
        self.map_size = 100
        self.grid = np.zeros((self.map_size, self.map_size), dtype=int)
        self.CELL_SIZE_CM = 2
        self.position = (self.map_size - 10, 10)
        self.heading = 135

        # Ultrasonic scan angles
        self.WIDE_SCAN_MIN = -80
        self.WIDE_SCAN_MAX = 80
        self.WIDE_STEP = 10

        self.set_steering(0)
        self.px.set_cam_pan_angle(0)
        time.sleep(0.2)

    def _process_detections(self, detection_result, image, timestamp_ms):
        """
        Callback from Mediapipe:
        - Only keep "stop sign" or "traffic light" with enough confidence.
        - Update fps every 10 frames.
        """
        if self.counter % self.fps_avg_frame_count == 0:
            self.fps = self.fps_avg_frame_count / (time.time() - self.start_time)
            self.start_time = time.time()

        self.detection_result_list.append(detection_result)
        self.counter += 1

        # Reset detection flags
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
                # minimal: treat traffic light as red if found
                self.red_light_detected = True
                self.current_detections.append(detection)

    def update_camera_and_detect(self):
        """
        Grab frame from picam2, run detection, draw bounding boxes.
        """
        frame = self.picam2.capture_array()
        cv2.putText(frame, f"FPS={self.fps:.1f}", (24,50),
                    cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),1,cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.detector.detect_async(mp_image, time.time_ns()//1_000_000)

        if self.detection_result_list:
            detection_result = self.detection_result_list.pop(0)
            frame = visualize_detection(frame, detection_result)

        cv2.imshow("Object Detection", frame)
        cv2.waitKey(1)

    def check_traffic_logic(self):
        """
        If stop_sign_detected => stop 3s
        If traffic light => also treat as red => stop 3s
        """
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
        """
        Average multiple ultrasonic readings
        """
        total = 0
        valid = 0
        for _ in range(samples):
            d = self.px.ultrasonic.read()
            if 0 <= d <= 200:
                total += d
                valid += 1
            time.sleep(0.03)
        if valid == 0:
            return 100
        return total / valid

    def scan_angle(self, angle, settle=0.2):
        self.px.set_cam_pan_angle(angle)
        time.sleep(settle)
        return self.read_ultrasonic(3)

    def wide_scan(self):
        """
        Full ultrasonic sweep to update the occupancy grid.
        """
        print("\n--- Starting wide scan ---")
        for angle in range(self.WIDE_SCAN_MIN, self.WIDE_SCAN_MAX + 1, self.WIDE_STEP):
            dist = self.scan_angle(angle)
            if dist < self.safe_distance:
                self._mark_obstacle(angle, dist)
        self.px.set_cam_pan_angle(0)
        self._print_map()

    def _mark_obstacle(self, sensor_angle, dist_cm):
        """
        Mark small zone around the detected distance in the grid.
        """
        if dist_cm >= self.safe_distance:
            return
        total_angle = self.heading + sensor_angle
        rad = math.radians(total_angle)
        for r in range(int(dist_cm - 5), int(dist_cm + 5), 2):
            if r < 0:
                continue
            gx = (math.cos(rad) * r) / self.CELL_SIZE_CM
            gy = (math.sin(rad) * r) / self.CELL_SIZE_CM
            px = int(self.position[0] + gx)
            py = int(self.position[1] + gy)
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                self.grid[py, px] = 1

    def _print_map(self):
        """
        Print the map of obstacles (1) vs free space (0).
        """
        RED = "\033[31m"
        GREEN = "\033[32m"
        RESET = "\033[0m"
        rows = np.any(self.grid == 1, axis=1)
        cols = np.any(self.grid == 1, axis=0)
        if not np.any(rows) or not np.any(cols):
            print("Map is empty - no obstacles.")
            return
        r_indices = np.where(rows)[0]
        c_indices = np.where(cols)[0]
        rmin, rmax = r_indices[0], r_indices[-1]
        cmin, cmax = c_indices[0], c_indices[-1]
        margin = 2
        rmin = max(rmin - margin, 0)
        rmax = min(rmax + margin, self.map_size - 1)
        cmin = max(cmin - margin, 0)
        cmax = min(cmax + margin, self.map_size - 1)
        print("\nOccupancy Grid (Red=obs, Green=clear):")
        subgrid = self.grid[rmin:rmax+1, cmin:cmax+1]
        for row in subgrid:
            line = "".join(f"{RED}1{RESET}" if c else f"{GREEN}0{RESET}" for c in row)
            print(line)

    def find_path(self, gx, gy):
        """
        A* pathfinding from self.position to (gx,gy).
        """
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
        moves = [(1,0),(-1,0),(0,1),(0,-1),
                 (1,1),(1,-1),(-1,1),(-1,-1)]

        while frontier:
            _, current = heappop(frontier)
            if current == goal:
                # Reconstruct
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
        """
        Quick left/center/right ultrasonic check.
        """
        left_d  = self.scan_angle(-80)
        center_d= self.scan_angle(0)
        right_d = self.scan_angle(80)
        self.px.set_cam_pan_angle(0)
        time.sleep(0.1)
        return {"left": left_d, "center": center_d, "right": right_d}

    def pick_best_direction(self, dists):
        """
        Decide direction to pivot or go straight based on safe distances.
        """
        good = {k:v for (k,v) in dists.items() if v >= self.safe_distance}
        if not good:
            return "none"
        if "center" in good:
            return "center"
        ld = good.get("left", -1)
        rd = good.get("right", -1)
        return "left" if ld >= rd else "right"

    def pivot_turn(self, direction, duration=1.2, forward=True):
        """
        Pivot turn left or right, then do a wide scan and re-run pathfinding.
        """
        if direction == "left":
            angle = -20
            print("Pivot turn LEFT")
        else:
            angle = 19
            print("Pivot turn RIGHT")
        self.set_steering(angle)
        if forward:
            self.forward(speed=self.slow_speed, duration=duration * 1.8)
        else:
            self.backward(speed=self.slow_speed, duration=duration)
        self.set_steering(0)

        # After a turn, do a wide scan => update map
        print("Turn complete => rescanning map.")
        self.wide_scan()

    def move_to_cell(self, x, y):
        """
        Move one cell, checking for obstacles, traffic signals, etc.
        If we back up => re-scan as well. Return True if we succeed, False if blocked.
        """
        print(f"\nMove to cell ({x},{y})")
        while True:
            # Run object detection + traffic logic each loop
            self.update_camera_and_detect()
            self.check_traffic_logic()

            dist = self.read_ultrasonic(samples=3)
            if dist >= self.safe_distance:
                print("Safe => forward 0.5s")
                self.set_steering(0)
                self.forward(duration=0.5)
                self.position = (x, y)
                return True
            elif dist < self.danger_distance:
                print("Too close => backing up => re-scan for new map.")
                self.backward(duration=1.0)
                self.wide_scan()  # Re-scan after backup
                return False
            else:
                print("Obstacle => scanning L/C/R.")
                triple = self.scan3()
                best = self.pick_best_direction(triple)
                if best == "none":
                    print("No path => back up => re-scan.")
                    self.backward(duration=1.0)
                    self.wide_scan()  # re-scan
                    return False
                elif best == "center":
                    print("Center => slow forward 0.5s")
                    self.forward(speed=self.slow_speed, duration=0.5)
                else:
                    # pivot turn => wide_scan inside pivot_turn
                    self.pivot_turn(best, duration=1.5, forward=True)
                    return False

            time.sleep(0.2)

    def navigate_to_goal(self, gx, gy):
        """
        - Do an initial wide scan
        - A* path => follow each cell
        - If we must pivot or back up => we do a new wide scan in move_to_cell/pivot_turn
        - Recompute path after each pivot/back up
        """
        print(f"\nPlanning path to goal=({gx},{gy})")
        self.wide_scan()
        while True:
            path = self.find_path(gx, gy)
            if not path:
                print("No valid path found!")
                return False
            print(f"Found path with {len(path)-1} steps.")
            cells = path[1:]
            # Follow each cell
            for cell in cells:
                success = self.move_to_cell(*cell)
                if not success:
                    # We pivoted or backed up => map changed => break to re-run path
                    break
                if cell == (gx, gy):
                    print("Reached final goal!")
                    self.stop()
                    return True
            # Re-loop => new pathfinding from updated map

    def run(self):
        """
        Example main routine: set a goal, then do navigate_to_goal.
        """
        GOAL_X = 84
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


if __name__=="__main__":
    main()
