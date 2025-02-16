#!/usr/bin/env python3
"""
Team IoT - CS437: Internet of Things
Lab 1B - Step 8: Smart Navigation with Advanced Mapping
Members: Mohammad Tamim, Jacob Fuehne, Jeffery Fan, KyoSook Shin
NetIDs: tamim2, jfuehne2, jfan16, kyosook2

Core capabilities:
- Basic hardware control (motors, servos, sensors)
- Movement functions (forward, backward, turning)
- Advanced mapping using ultrasonic scanning
- A* path planning with obstacle avoidance
"""

from picarx import Picarx
import time
import math
import numpy as np
from heapq import heappush, heappop
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from picamera2 import Picamera2
from utils import visualize
import asyncio

class TeamIoT_SmartNavigator:
    def __init__(self):
        # Main interface to control the car
        self.px = Picarx()

        # Initialize camera
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (640, 480)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()

        # Visualization parameters
        self.row_size = 50  # pixels
        self.left_margin = 24  # pixels
        self.text_color = (0, 0, 0)  # black
        self.font_size = 1
        self.font_thickness = 1
        self.fps_avg_frame_count = 10
        self.counter = 0
        self.fps = 0
        self.start_time = time.time()
        self.detection_frame = None
        self.detection_result_list = []

        # Initialize object detection
        base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=5,
            score_threshold=0.5,
            result_callback=self._process_detection
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.current_detections = []
        self.stop_sign_detected = False
        self.last_stop_time = 0
        self.STOP_SIGN_WAIT = 2.0  # Wait time in seconds at stop signs


        # Movement speeds (tweak these values during testing)
        self.forward_speed = 10       # Regular cruising speed; adjust as needed.
        self.slow_speed = 10          # Reduced speed for maneuvers/turns.
        self.backup_speed = 8         # Reverse speed for backing up safely.

        # Distance thresholds (in centimeters)
        self.safe_distance = 43       # Distance considered safe to move forward.
        self.danger_distance = 19     # Distance triggering a backup maneuver.
        self.very_danger_distance = 15  # Reserved for future use; not active here.

        # Steering and sensor calibration
        ### Do not use servo_offset, instead, use `sudo python3 calibration/calibration.py`
        # keeping this here in case extra offset is needed by someone's kit.
        self.servo_offset = 0      # Calibration offset for the steering servo.
                                     # Adjust this if the car doesn't drive straight.
        
        # Occupancy grid settings
        self.map_size = 100           # Grid dimensions: 100 x 100 cells. #mostly arbitrary
        self.grid = np.zeros((self.map_size, self.map_size), dtype=int)
        # self.CELL_SIZE_CM = 2         # Each grid cell represents 2 cm in the real world.
        self.CELL_SIZE_CM = 2.54        # we uses inches in 'merica.  1 inch = 2.54 cm
                                        # (using this so we can map the course with a tape measure)
                                     # Tweak this if your mapping resolution needs to change.
        self.wheelbase = 8.89 #3.5 inches / 8.89 cm between axles of the Picar-X 
        # Initial position and orientation
        # self.position = (self.map_size - 10, 10)  # Starting cell (near bottom-right).
        # self.position = (10, 10)
        self.position = (55, 90)
        # self.heading = 135            # Starting heading in degrees (facing top-left).
        self.heading = 0
        self.goal_position = (55, 10)
                                     # Adjust if the car's initial orientation changes.

        # Ultrasonic sensor scan settings
        self.WIDE_SCAN_MIN = -90      # Leftmost scan angle (degrees).
        self.WIDE_SCAN_MAX = 90       # Rightmost scan angle (degrees).
        self.WIDE_STEP = 10           # Angle increment for wide scanning.
                                     # Modify these if a wider or narrower scan is needed.

        # Add monitoring control flags
        self.monitoring_task = None
        self.is_moving = False


    def initialize_servos(self):
        """Initialize the servos and wheels to 0.  Seperate from set_steering because we need a reset that's not async

        Args:
            angle (int): Desired steering angle. The angle is clamped between -33 and 33.
                         If the car is not steering straight, adjust the servo_offset.
        """
        angle = 0 # set angles to 0
        # Adjust the servo angle with the calibration offset.
        self.px.set_dir_servo_angle(angle + self.servo_offset) 
        time.sleep(0.2)  # Allow time for the steering to physically adjust.
        self.px.set_cam_pan_angle(0)  # Set sensor to face forward.
        time.sleep(0.2)               # Allow servos to settle.

    async def set_steering(self, angle):
        """Fine control over our steering.

        Args:
            angle (int): Desired steering angle. The angle is clamped between -33 and 33.
                         If the car is not steering straight, adjust the servo_offset.
        """
        angle = max(-33, min(33, angle))  # Ensure the angle is within safe limits.
        # Adjust the servo angle with the calibration offset.
        self.px.set_dir_servo_angle(angle + self.servo_offset)
        time.sleep(0.2)  # Allow time for the steering to physically adjust.


    def _process_detection(self, detection_result, image, timestamp_ms):
        """Process detection results and update visualization."""
        if self.counter % self.fps_avg_frame_count == 0:
            self.fps = self.fps_avg_frame_count / (time.time() - self.start_time)
            self.start_time = time.time()

        self.detection_result_list.append(detection_result)
        self.counter += 1
        
        self.current_detections = []
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.category_name == "stop sign" and category.score > 0.5:
                self.stop_sign_detected = True
                self.current_detections.append(detection)


    def check_for_stop_sign(self):
        """Capture and process image for stop sign detection."""
        self.update_display()  # Update the visual feed
        
        if self.stop_sign_detected:
            current_time = time.time()
            if current_time - self.last_stop_time > self.STOP_SIGN_WAIT * 2:
                print("Stop sign detected! Stopping for 2 seconds...")
                self.stop()
                time.sleep(self.STOP_SIGN_WAIT)
                self.last_stop_time = current_time
                self.stop_sign_detected = False
                return True
        return False

    async def forward(self, speed=None, duration=None):
        """Drive forward, optionally for a specific time.

        Args:
            speed (int, optional): Speed at which to drive forward.
                                   Defaults to forward_speed.
            duration (float, optional): Duration (in seconds) for forward motion.
                                        Useful for moving exactly one grid cell.
        """
        if speed is None:
            speed = self.forward_speed
            
        # Start the monitoring before moving
        await self.start_monitoring()
        
        # Start moving
        self.px.forward(speed)
        
        try:
            if duration:
                # Check monitoring_task status periodically
                start_time = time.time()
                while time.time() - start_time < duration:
                    if not self.is_moving or self.monitoring_task.done():
                        # Monitoring task detected an obstacle and stopped
                        time_elapsed = time.time() - start_time
                        distance = speed * time_elapsed ###### THIS MIGHT NEEDS ADJUSTING!!! #############
                        self.update_position(distance)
                        return False
                    await asyncio.sleep(0.05)  # Short sleep to allow other tasks to run
                distance = speed * duration * 0.8  # 0.8 is a correction factor
                self.update_position(distance)
                return True
        finally:
            self.stop()
            self.stop_monitoring()

    async def backward(self, speed=None, duration=0.7):
        """Back up, usually when we're too close to an obstacle.

        Args:
            speed (int, optional): Speed for backing up.
                                   Defaults to backup_speed.
            duration (float, optional): Time to back up (seconds).
                                        Adjust if more/less backing up is needed.
        """
        if speed is None:
            speed = self.backup_speed
        self.px.backward(speed)
        await asyncio.sleep(duration)
        distance = -speed * duration
        self.update_position(distance)
        self.stop()

    def stop(self):
        """Come to a full stop."""
        self.px.forward(0)
        self.stop_monitoring()

    async def read_ultrasonic(self, samples=5):
        """Get accurate distance reading by averaging multiple samples.

        Args:
            samples (int, optional): Number of sensor readings to average.
        Returns:
            float: Averaged distance reading.
        """
        total = 0
        count = 0
        for _ in range(samples):
            d = self.px.ultrasonic.read()
            # print(d)
            if 0 <= d <= 200:  # Only consider valid readings.
                total += d
                count += 1
            await asyncio.sleep(0.03)
        if count == 0:
            return 100  # Default fallback if no valid reading.
        return total / count
    
    def _process_detection(self, detection_result, image, timestamp_ms):
        """Process detection results and update visualization."""
        if self.counter % self.fps_avg_frame_count == 0:
            self.fps = self.fps_avg_frame_count / (time.time() - self.start_time)
            self.start_time = time.time()

        self.detection_result_list.append(detection_result)
        self.counter += 1
        
        self.current_detections = []
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.category_name == "stop sign" and category.score > 0.5:
                self.stop_sign_detected = True
                self.current_detections.append(detection)

    def update_display(self):
        """Update and show the camera feed with object detection visualization."""
        image = self.picam2.capture_array()
        image = cv2.resize(image, (640, 480))
        # image = cv2.flip(image, -1)

        # Add FPS counter
        fps_text = f'FPS = {self.fps:.1f}'
        cv2.putText(image, fps_text, (self.left_margin, self.row_size),
                   cv2.FONT_HERSHEY_DUPLEX, self.font_size, self.text_color,
                   self.font_thickness, cv2.LINE_AA)

        # Process image for object detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Draw detection results
        if self.detection_result_list:
            image = visualize(image, self.detection_result_list[0])
            self.detection_frame = image
            self.detection_result_list.clear()

        # Display the frame
        if self.detection_frame is not None:
            cv2.imshow('Smart Navigator - Object Detection', self.detection_frame)
        cv2.waitKey(1)

    async def ultrasonic_monitor(self):
        """
        Continously check the ultrasonic monitor to see what's in front of us so we stop running into walls or objects.  Coincidentally, this also helps handle "dynamic objects" better.
        """
        while self.is_moving:
            try:
                # print("monitoring...")
                distance = await self.read_ultrasonic(samples=2)  # Reduced samples for faster response
                if distance < self.danger_distance:
                    print("danger")
                    self.stop()
                    self.is_moving = False
                    break
                await asyncio.sleep(0.05)  # Small delay between readings
            except Exception as e:
                print(f"Error in ultrasonic monitor: {e}")
                self.stop()
                self.is_moving = False
                break

    async def start_monitoring(self):
        """Start the background ultrasonic monitoring."""
        if not self.monitoring_task:
            self.is_moving = True
            self.monitoring_task = asyncio.create_task(self.ultrasonic_monitor())

    def stop_monitoring(self):
        """Stop the background ultrasonic monitoring."""
        self.is_moving = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
    
    async def scan_angle(self, angle, settle=0.2):
        """Point the sensor in a specific direction and get a distance reading.

        Args:
            angle (int): Angle in degrees to which the sensor is directed.
            settle (float, optional): Delay for sensor to stabilize.
        Returns:
            float: Measured distance at that angle.
        """
        self.px.set_cam_pan_angle(angle)
        await asyncio.sleep(settle)  # Wait for the sensor to settle.
        return await self.read_ultrasonic(samples=3)

    async def scan3(self):
        """Quick check: measure distances to the left, center, and right.

        Returns:
            dict: Distance readings for keys "left", "center", and "right".
        """
        left_d = await self.scan_angle(-80)
        center_d = await self.scan_angle(0)
        right_d = await self.scan_angle(80)
        self.px.set_cam_pan_angle(0)  # Reset sensor to face forward.
        await asyncio.sleep(0.1)
        self.print_map()
        return {"left": left_d, "center": center_d, "right": right_d}

    async def wide_scan(self, step_size=1, sweep_delay=0.0005):
        """
        Build a map of our surroundings by scanning in a wide arc.
        
        The sensor sweeps from left to right. For each angle, if the distance
        is less than our safe threshold, the corresponding grid cells are marked
        as obstacles. Finally, the updated map is visualized.
        """
        print("\n--- Starting Wide Area Scan ---")
        # Sweep from left to right
        for angle in range(self.WIDE_SCAN_MIN, self.WIDE_SCAN_MAX + 1, step_size):
            self.px.set_cam_pan_angle(angle)  # Move the servo to the desired angle
            await asyncio.sleep(sweep_delay)  # Allow time for movement stabilization
            
            dist = await self.read_ultrasonic(samples=3)  # Take distance readings
            
            if dist < self.safe_distance:
                self.mark_obstacle(angle, dist)  # Mark obstacle in the map

        # Return servo to the center position
        self.px.set_cam_pan_angle(0)
        self.print_map()

    def calculate_new_heading(self, turn_direction, steering_angle, duration, speed):
        """
        Calculate the new heading after a turn based on steering angle, duration, and speed.
        
        Args:
            turn_direction (str): "left" or "right"
            steering_angle (float): Angle of the wheels in degrees
            duration (float): Duration of the turn in seconds
            speed (float): Speed of the car during the turn
            
        Returns:
            float: New heading in degrees (0-360)
        """
        # Constants for turn radius calculation
        # Distance between front and rear axles in cm

        
        # Calculate turn radius (refer: Ackermann Steering Geometry formula)
        turn_radius = abs(self.wheelbase / math.tan(math.radians(steering_angle)))
        
        # Calculate distance traveled during turn
        distance = speed * duration
        
        # Calculate angle changed during turn (arc length / radius)
        angle_change = math.degrees(distance / turn_radius)
        
        # Adjust angle based on turn direction
        if turn_direction == "left":
            new_heading = (self.heading + angle_change) % 360
        else:  # right turn
            new_heading = (self.heading - angle_change) % 360
            
        return new_heading

    def update_position(self, movement_type, duration, speed, steering_angle=0):
        """
        Update position based on movement type and parameters.
        
        Args:
            movement_type (str): "forward", "backward", or "turn"
            duration (float): Duration of movement in seconds
            speed (float): Speed of movement
            steering_angle (float): Angle of steering for turns (degrees)
        """
        # Convert speed from arbitrary units to cm/s
        SPEED_TO_CM = 15.0  # Calibration factor: speed of 10 ≈ 15 cm/s
        distance = (speed * SPEED_TO_CM * duration)  # Distance in cm
        
        if movement_type in ["forward", "backward"]:
            # For straight movements, use heading to calculate new position
            rad_heading = math.radians(self.heading)
            dx = distance * math.cos(rad_heading)
            dy = distance * math.sin(rad_heading)
            
            # Reverse direction if backing up
            if movement_type == "backward":
                dx = -dx
                dy = -dy
                
        elif movement_type == "turn":
            # Calculate turn radius
            turn_radius = abs(self.wheelbase / math.tan(math.radians(steering_angle))) if steering_angle != 0 else float('inf')
            
            # Calculate angle traversed in radians
            angle_traversed = distance / turn_radius if turn_radius != float('inf') else 0
            
            # Calculate change in position based on arc movement
            if steering_angle > 0:  # Right turn
                angle_traversed = -angle_traversed
            
            # Initial heading in radians
            initial_heading_rad = math.radians(self.heading)
            
            # Calculate position change using arc formulas
            if turn_radius != float('inf'):
                dx = turn_radius * (math.sin(initial_heading_rad + angle_traversed) - math.sin(initial_heading_rad))
                dy = turn_radius * (math.cos(initial_heading_rad) - math.cos(initial_heading_rad + angle_traversed))
            else:
                # Straight line movement if turn_radius is infinite
                dx = distance * math.cos(initial_heading_rad)
                dy = distance * math.sin(initial_heading_rad)
        
        # Convert position change from cm to grid cells
        dx_cells = dx / self.CELL_SIZE_CM
        dy_cells = dy / self.CELL_SIZE_CM
        
        # Update position
        new_x = self.position[0] + dx_cells
        new_y = self.position[1] + dy_cells
        
        # Ensure we stay within grid bounds
        new_x = max(0, min(new_x, self.map_size - 1))
        new_y = max(0, min(new_y, self.map_size - 1))
        
        self.position = (new_x, new_y)
        print(f"Updated position: ({new_x:.1f}, {new_y:.1f})")
    
        # def update_position(self, distance, heading_angle=None):
        # """
        # Update the car's position based on distance traveled and heading.
        
        # Args:
        #     distance (float): Distance traveled in cm (negative for reverse)
        #     heading_angle (float, optional): Override current heading if provided
        # """
        # # Use provided heading if available, otherwise use current heading
        # angle = heading_angle if heading_angle is not None else self.heading
        
        # # Convert heading to radians
        # rad = math.radians(angle)
        
        # # Calculate position changes in grid coordinates
        # dx = (distance * math.cos(rad)) / self.CELL_SIZE_CM
        # dy = (distance * math.sin(rad)) / self.CELL_SIZE_CM
        
        # # Update position
        # new_x = self.position[0] + dx
        # new_y = self.position[1] + dy
        
        # # Ensure we stay within grid bounds
        # new_x = max(0, min(self.map_size - 1, new_x))
        # new_y = max(0, min(self.map_size - 1, new_y))
        
        # self.position = (new_x, new_y)
        # print(f"Updated position: ({new_x:.1f}, {new_y:.1f}), Heading: {angle:.1f}°")
    def mark_obstacle(self, sensor_angle, dist_cm):
        """
        Record an obstacle on our map with a safety buffer.

        The obstacle is marked not only at the detected distance, but in a small range
        (±5 cm) around it. This accounts for the car's size and increases safety.

        Args:
            sensor_angle (int): Angle at which the obstacle is detected.
            dist_cm (float): Distance reading in centimeters.
        """

        total_angle = self.heading + sensor_angle  # Combine car's heading with sensor angle
        rad = math.radians(total_angle)
        print("marking obstacles")
        # First mark the clear path up to the obstacle (or full path if no obstacle)
        safe_dist = min(dist_cm, self.safe_distance)
        for r in range(0, int(safe_dist), 2):
            # Convert polar coordinates to grid coordinates
            gx = (math.cos(rad) * r) / self.CELL_SIZE_CM
            gy = (math.sin(rad) * r) / self.CELL_SIZE_CM
            px = int(self.position[0] + gx)
            py = int(self.position[1] + gy)
            
            # Mark as safe if within grid bounds
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                self.grid[py, px] = 0
        
        # If we detected an obstacle within safe distance, mark it and its buffer zone
        if dist_cm < self.safe_distance:
            # Mark the obstacle and a buffer zone (±5 cm) around it
            for r in range(int(dist_cm - 5), int(dist_cm + 5), 2):
                if r < 0:
                    continue
                gx = (math.cos(rad) * r) / self.CELL_SIZE_CM
                gy = (math.sin(rad) * r) / self.CELL_SIZE_CM
                px = int(self.position[0] + gx)
                py = int(self.position[1] + gy)
                
                if 0 <= px < self.map_size and 0 <= py < self.map_size:
                    self.grid[py, px] = 1

    # def print_map(self):
    #     """
    #     Display the current environment map using ANSI colors.
        
    #     - Red ("1") indicates an obstacle.
    #     - Green ("0") indicates a clear space.
        
    #     This visualization helps to debug and fine-tune the mapping process.
    #     """
    #     RED = "\033[31m"    # Red color for obstacles.
    #     GREEN = "\033[32m"  # Green color for clear paths.
    #     RESET = "\033[0m"   # Reset color to default.
        
    #     rows = np.any(self.grid == 1, axis=1)
    #     cols = np.any(self.grid == 1, axis=0)
    #     if not np.any(rows) or not np.any(cols):
    #         print("Map is empty - no obstacles detected")
    #         return
    #     r_indices = np.where(rows)[0]
    #     c_indices = np.where(cols)[0]
    #     rmin, rmax = r_indices[0], r_indices[-1]
    #     cmin, cmax = c_indices[0], c_indices[-1]
    #     # Add a margin around the detected area.
    #     margin = 2
    #     rmin = max(rmin - margin, 0)
    #     rmax = min(rmax + margin, self.map_size - 1)
    #     cmin = max(cmin - margin, 0)
    #     cmax = min(cmax + margin, self.map_size - 1)
        
    #     print("\nCurrent Environment Map (Red=obstacle, Green=clear):")
    #     for row in self.grid[rmin:rmax+1, cmin:cmax+1]:
    #         print("".join(f"{RED}1{RESET}" if c else f"{GREEN}0{RESET}" for c in row))

    def print_map(self):
        """
        Display the complete environment map using ANSI colors.
        
        - Red ("1") indicates an obstacle
        - Green ("0") indicates a clear space
        - Blue ("P") indicates the current position
        - Yellow ("G") indicates the goal position
        """
        RED = "\033[31m"     # Red color for obstacles
        GREEN = "\033[32m"   # Green color for clear paths
        BLUE = "\033[34m"    # Blue color for current position
        YELLOW = "\033[33m"  # Yellow color for goal position
        RESET = "\033[0m"    # Reset color to default
        
        print("\nCurrent Environment Map:")
        print(f"{RED}1{RESET}=obstacle, {GREEN}0{RESET}=clear, {BLUE}P{RESET}=position, {YELLOW}G{RESET}=goal")
        
        # Convert position to integer coordinates for marking
        pos_x, pos_y = int(round(self.position[0])), int(round(self.position[1]))
        
        # Get goal position from navigation target if it exists
        if self.goal_position:
            goal_x, goal_y = self.goal_position
        else:
            goal_x, goal_y = (-1,-1)
        
        # Print column numbers (coordinates) at the top
        print("   ", end="")
        for i in range(0, self.map_size, 10):
            print(f"{i:10}", end="")
        print()
        
        # Print the map with row numbers
        for y in range(self.map_size):
            # Print row number
            print(f"{y:2} ", end="")
            
            for x in range(self.map_size):
                if x == goal_x and y == goal_y:
                    print(f"{YELLOW}G{RESET}", end="")
                elif x == pos_x and y == pos_y:
                    print(f"{BLUE}P{RESET}", end="")
                elif self.grid[y, x] == 1:
                    print(f"{RED}1{RESET}", end="")
                else:
                    print(f"{GREEN}0{RESET}", end="")
            
            # Print row number again at the end
            print(f" {y:2}")
        
        # Print column numbers at the bottom
        print("   ", end="")
        for i in range(0, self.map_size, 10):
            print(f"{i:10}", end="")
        print()

    def find_path(self, gx, gy):
        """
        Compute the best path to the goal using the A* search algorithm.

        This function:
          - Validates the goal cell (ensuring it's not an obstacle).
          - Uses eight-directional movement to compute the path.
          - Returns a list of grid cells from the current position to the goal.
        
        Args:
            gx (int): Goal cell x-coordinate.
            gy (int): Goal cell y-coordinate.
        Returns:
            list or None: The computed path as a list of grid cells, or None if blocked.
        """
        start = self.position
        goal = (gx, gy)
        self.goal_position = goal

        if self.grid[goal[1], goal[0]] == 1:
            return None  # The goal cell is blocked.

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (-1, 1), (1, -1), (-1, -1)]
        while frontier:
            _, current = heappop(frontier)
            if current == goal:
                # Reconstruct the path from goal to start.
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            for dx, dy in moves:
                nx = current[0] + dx
                ny = current[1] + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.grid[ny, nx] == 0:  # Only travel through clear cells.
                        new_cost = cost_so_far[current] + 1
                        if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                            cost_so_far[(nx, ny)] = new_cost
                            priority = new_cost + heuristic((nx, ny), goal)
                            heappush(frontier, (priority, (nx, ny)))
                            came_from[(nx, ny)] = current
        return None


    def pick_best_direction(self, dists, target=None):
        """
        Decide which direction to steer based on sensor readings.

        Strategy:
          1. Prefer going straight if it is safe.
          2. If both left and right are safe and a target is provided,
             bias toward the target.
          3. Otherwise, choose the direction with more clearance.
          4. If no safe direction exists, return "none".
        
        Args:
            dists (dict): Readings for "left", "center", and "right".
            target (tuple, optional): Target grid cell (x,y).
        Returns:
            str: "left", "center", "right", or "none".
        """
        candidates = {k: v for k, v in dists.items() if v >= self.safe_distance}
        if not candidates:
            return "none"
        if "center" in candidates:
            cdist = candidates["center"]
            ldist = candidates.get("left", -1)
            rdist = candidates.get("right", -1)
            if cdist >= ldist and cdist >= rdist:
                return "center"
        if target is not None and "left" in candidates and "right" in candidates:
            cx, cy = self.position
            tx, ty = target
            return "left" if tx < cx else "right"
        if candidates.get("left", -1) > candidates.get("right", -1):
            return "left"
        elif candidates.get("right", -1) > candidates.get("left", -1):
            return "right"
        else:
            return "left"  # Default to left if uncertain.

    async def pivot_turn(self, direction, duration=1.2, forward=True):
        """
        Execute a sharp pivot turn to avoid obstacles.

        This maneuver uses a fixed steering angle for a brief period,
        then resets to straight. You can adjust the fixed angles if the car
        does not turn as expected.

        Args:
            direction (str): "left" or "right".
            duration (float, optional): Base duration for the turn.
            forward (bool, optional): Pivot while moving forward if True;
                                      otherwise, pivot while backing up.
        """
        if direction == "left":
            steering_angle = -30  # Tune this value for a sharper or gentler left turn.
            print("  Turning LEFT to avoid obstacle")
        else:
            steering_angle = 30   # Tune this value for a sharper or gentler right turn.
            print("  Turning RIGHT to avoid obstacle")
        await self.set_steering(steering_angle)

        # Calculate turn parameters
        turn_radius = abs(self.wheelbase / math.tan(math.radians(abs(steering_angle))))
        
        # Execute the turn
        speed = self.slow_speed
        if forward:
            await self.forward(speed=self.slow_speed, duration=duration * 1.8)
        else:
            await self.backward(speed=self.slow_speed, duration=duration * 1.0)
            speed = -speed #don't actually send negatives to self.backward(), for math only
        
        arc_distance = speed * duration * (1.8 if forward else 1.0)

        self.heading = self.calculate_new_heading(
            direction, 
            abs(steering_angle),
            duration * (1.8 if forward else 1.0),
            speed
        )
        self.update_position(arc_distance, self.heading)
        print(f"  New heading: {self.heading:.1f}°")

        await self.set_steering(0)  # Reset steering to center after the turn.


    async def move_to_cell(self, x, y):
        """
        Navigate cell-by-cell toward a target grid cell.

        The function:
          - Continuously checks for obstacles.
          - Moves forward if the path is clear.
          - Backs up if too close to an obstacle.
          - Scans and pivots if an obstacle is detected.
        Updates the internal position once the target cell is reached.

        Args:
            x (int): Target cell x-coordinate.
            y (int): Target cell y-coordinate.
        """
        print(f"\nNavigating to position ({x}, {y})")
        # backup_attempts = 0
        # distance_from_object_in_front = await self.read_ultrasonic()

        while True:
            # distance_from_object_in_front = await self.read_ultrasonic()
            # print(distance_from_object_in_front)
            if self.check_for_stop_sign():
                print("Resuming navigation after stop sign...")
            
            if await self.forward(speed=self.forward_speed, duration=0.5):
                # If we complete the forward motion without emergency stop
                self.position = (x, y)
                break
            else:
                print("Too close! Backing up to avoid collision...")
                await self.backward(duration=0.5)
            # # If the path is clear, move forward one cell.
            # if dist >= self.safe_distance:
            #     print("Path clear, moving ahead...")
            #     self.set_steering(0)
            #     self.forward(speed=self.forward_speed, duration=0.5)  # Duration tuned for one cell's length.
            #     self.position = (x, y)  # Update our internal map position.
            #     break

            # # If too close to an obstacle, back up.
            # if dist < self.danger_distance:
            #     print("Too close! Backing up to avoid collision...")
            #     self.backward(duration=1.0)  # Adjust backing up duration if necessary.
            #     backup_attempts += 1
            #     continue

            # Otherwise, scan for alternative routes.
            print("Obstacle detected - scanning alternative routes...")
            scans = await self.scan3()
            print(f"Distance readings: {scans}")
            best_dir = self.pick_best_direction(scans, target=(x, y))

            if best_dir == "none":
                print("No clear path found - backing up further...")
                await self.backward(duration=1.0)
            elif best_dir == "center":
                print("Center path clear - proceeding cautiously...")
                await self.forward(speed=self.slow_speed, duration=0.8)
            elif best_dir == "left":
                print("Pivot turning left to avoid obstacle...")
                await self.pivot_turn("left", duration=1.5, forward=True)
            elif best_dir == "right":
                print("Pivot turning right to avoid obstacle...")
                await self.pivot_turn("right", duration=1.5, forward=True)

            await asyncio.sleep(0.2)  # Small pause between maneuvers.

    async def navigate_to_goal(self, gx, gy):
        """
        Plan and execute a complete route to the goal cell.

        Steps:
          1. Update the map via a wide scan.
          2. Compute a path from the current position to the goal using A*.
          3. Follow the computed path cell-by-cell while avoiding obstacles.
          4. Stop once the goal cell is reached.

        Args:
            gx (int): Goal cell x-coordinate.
            gy (int): Goal cell y-coordinate.
        Returns:
            bool: True if the goal is reached; False otherwise.
        """
        print(f"\nPlanning route to goal ({gx}, {gy})...")
        try:
            await self.wide_scan()
            path = self.find_path(gx, gy)
            if not path:
                print("No path found!")
                return False

            cells = path[1:]
            for idx, cell in enumerate(cells):
                await self.move_to_cell(*cell)
                self.update_display()  # Update display during navigation
                if cell == (gx, gy):
                    print("\nFinal cell reached (goal). Stopping.")
                    self.stop()
                    return True

            print("\nArrived at goal!")
            return True
            
        except KeyboardInterrupt:
            print("\nNavigation stopped by user")
            return False
        finally:
            cv2.destroyAllWindows()  # Clean up display windows



async def main():
    """Start up our smart navigation system."""
    try:
        print("\n=== Team IoT - Smart Navigation System ===")
        nav = TeamIoT_SmartNavigator()
        nav.initialize_servos()
        await nav.navigate_to_goal(55, 10)  # Goal cell coordinates (adjust as needed)
    except KeyboardInterrupt:
        print("\nNavigation stopped by user")
    finally:
        nav.stop()
        print("System shutdown complete")

asyncio.run(main())