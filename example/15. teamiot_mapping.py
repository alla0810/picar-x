#!/usr/bin/env python3
# Team IoT - CS437: Internet of Things
# Lab 1 Part 1 Submission
# Members: Mohammad Tamim, Jacob Fuehne, Jeffery Fan, KyoSook Shin
# NetIDs: tamim2, jfuehne2, jfan16, kyosook2

import time
import numpy as np
import math

class PicarXMapping:
    def __init__(self, picar, map_size = 100):
        self.px = picar # Picar-X object
        self.map_size = map_size
        self.grid = np.zeros((map_size, map_size), dtype=int)
        self.position = (map_size // 2, map_size // 2) # Set the starting position to the center
        self.angle = 0 # Current vehicle direction (0 degrees: forward)
        self.CAM_PAN_MIN = -60 # min camera angle
        self.CAM_PAN_MAX = 60

    def check_distance(self):
        """ Measure distance using the ultrasonic sensor"""
        d = self.px.ultrasonic.read()
        return d if 0 <= d <= 200 else 100 # Default to 100cm if out of range
    
    def set_cam_pan_angle(self, value):
        """ Set the camera pan (left-right) angle using the library function """
        self.px.set_cam_pan_angle(value)

    def update_map(self, angle_step=5):
        """ Scan surroundings and update the map """
        for angle in range(self.CAM_PAN_MIN, self.CAM_PAN_MAX + 1, angle_step):
            self.set_cam_pan_angle(angle)
            distance = self.check_distance()
            self.mark_obstacle(angle, distance)
        self.print_map()
        self.set_cam_pan_angle(0)        

    def mark_obstacle(self, angle, distance):
        """ Mark obstacle location on the map """
        if distance < 200: # Record only within 200cm
            x_offset = int(math.cos(math.radians(angle)) * distance)
            y_offset = int(math.sin(math.radians(angle)) * distance)
            x, y = self.position[0] + x_offset, self.position[1] + y_offset
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.grid[y, x] = 1 # Mark obstacle

    def interpolate_map(self):
        """ Interpolate distance data """
        for x in range(self.map_size - 1):
            for y in range(self.map_size - 1):
                if self.grid[y,x] == 1 and self.grid[y, x+1] == 1:
                    self.grid[y, x+1] = 1 # Simple interpolation

    def move_vehicle(self, velocity, time_interval=5):
        """ Move the vehicle forward based on velocity and time interval """
        self.px.forward(velocity)
        time.sleep(time_interval)
        self.px.stop()

        """ Update vehicle position based on velocity data """
        dx = int(math.cos(math.radians(self.angle)) * velocity * time_interval)
        dy = int(math.sin(math.radians(self.angle)) * velocity * time_interval)
        new_x, new_y = self.position[0] + dx, self.position[1] + dy

        if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
            self.position = (new_x, new_y)  # Update vehicle position

    def run_mapping(self, velocity, time_interval=5):
        """ Continuously generate the map while moving the vehicle """
        while True:
            start_time = time.time()

            self.update_map()
            self.interpolate_map()
            self.move_vehicle(velocity, time_interval)

            elapsed_time = time.time() - start_time
            sleep_time = max(0, time_interval - elapsed_time) 
            time.sleep(sleep_time)

    def print_map(self):
        """ Print the current map representation """
        RED = "\033[41m  \033[0m"  # Red background for obstacles (1)
        WHITE = "\033[47m  \033[0m" # White background for empty space (0)

        print("\nCurrent Map:")
        for row in self.grid:
            print("".join(RED if cell == 1 else WHITE for cell in row))

if __name__ == "__main__":
    from picarx import Picarx
    px = Picarx()
    mapper = PicarXMapping(px)
    mapper.run_mapping(velocity=10, time_interval=5)


