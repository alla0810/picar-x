#!/usr/bin/env python3
# Team IoT - CS437: Internet of Things
# Lab 1 Part 1 Submission
# Members: Mohammad Tamim, Jacob Fuehne, Jeffery Fan, KyoSook Shin
# NetIDs: tamim2, jfuehne2, jfan16, kyosook2

from picarx import Picarx
import time

class TeamIoT_ThreeAngleAvoider:
    def __init__(self):
        """Initialize PiCar-X and obstacle avoidance parameters."""
        self.px = Picarx()

        # Speed settings
        self.forward_speed = 15  # Normal driving speed
        self.slow_speed = 10     # Cautious turning speed
        self.backup_speed = 8    # Reverse speed
        
        # Distance thresholds (cm)
        self.safe_distance = 37   # Continue forward if clear
        self.danger_distance = 27 # Immediate backup if too close
        
        # Steering calibration
        self.servo_offset = -6.1  # Adjust for hardware drift
        
        # Reset steering and sensor to default position
        self.set_steering(0)
        self.px.set_cam_pan_angle(0)
        time.sleep(0.5)

    def set_steering(self, angle):
        """Adjust steering angle with offset correction."""
        self.px.set_dir_servo_angle(angle + self.servo_offset)

    def forward(self, speed=None, duration=None):
        """Move forward with optional duration."""
        if speed is None:
            speed = self.forward_speed
        self.px.forward(speed)
        if duration:
            time.sleep(duration)
            self.stop()

    def backward(self, speed=None, duration=1.0):
        """Move backward for a specified duration."""
        if speed is None:
            speed = self.backup_speed
        self.px.backward(speed)
        time.sleep(duration)
        self.stop()

    def stop(self):
        """Stop all movement."""
        self.px.forward(0)

    def check_distance(self):
        """Measure distance using the ultrasonic sensor."""
        d = self.px.ultrasonic.read()
        return d if 0 <= d <= 200 else 100  # Default to 100cm if reading is unreliable

    def scan_angle(self, angle, settle=0.2):
        """Point sensor at a specific angle and measure distance."""
        self.px.set_cam_pan_angle(angle)
        time.sleep(settle)  # Allow sensor to stabilize
        return self.check_distance()

    def scan3(self):
        """Scan left (-80°), center (0°), and right (+80°)."""
        print("Scanning surroundings...")
        left_d = self.scan_angle(-80)
        center_d = self.scan_angle(0)
        right_d = self.scan_angle(80)

        # Reset camera position after scanning
        self.px.set_cam_pan_angle(0)
        time.sleep(0.1)

        return {"left": left_d, "center": center_d, "right": right_d}

    def pick_best_direction(self, dists):
        """Choose the best direction based on scan results."""
        candidates = {k: v for k, v in dists.items() if v >= self.safe_distance}

        if not candidates:
            return "none"  # No clear path

        if "center" in candidates and candidates["center"] >= max(candidates.values()):
            return "center"

        left = candidates.get("left", -1)
        right = candidates.get("right", -1)

        return "left" if left >= right else "right"

    def pivot_turn(self, direction, duration=1.2):
        """Execute a turn in the chosen direction."""
        print(f"Turning {direction.upper()}...")
        angle = -99 if direction == "left" else 90  # Defined turn angles
        self.set_steering(angle)
        self.forward(self.slow_speed, duration * 1.3)
        self.set_steering(0)  # Straighten out after turn

    def run(self):
        """Main obstacle avoidance loop."""
        print("\n=== Team IoT - Obstacle Avoidance System ===")
        print(f"Safe distance: {self.safe_distance}cm, Danger threshold: <{self.danger_distance}cm")
        print("Starting system...\n")

        try:
            while True:
                dist = self.check_distance()
                print(f"Front distance: {dist}cm", end='\r')

                if dist >= self.safe_distance:
                    self.set_steering(0)  # Keep straight
                    self.forward(duration=0.5)
                    continue

                if dist < self.danger_distance:
                    print("\nObstacle detected—reversing")
                    self.backward(duration=0.5)

                    if self.check_distance() < self.danger_distance:
                        print("Still too close—backing up further")
                        self.backward(duration=0.5)

                    scans = self.scan3()
                    direction = self.pick_best_direction(scans)

                    if direction == "none":
                        print("No clear path—reversing again")
                        self.backward(duration=1.0)
                    else:
                        self.pivot_turn(direction)

                else:
                    print("\nCaution: scanning surroundings")
                    self.stop()
                    scans = self.scan3()
                    direction = self.pick_best_direction(scans)

                    if direction == "none":
                        print("No clear path—backing up")
                        self.backward(duration=1.2)
                    else:
                        self.pivot_turn(direction)

        except KeyboardInterrupt:
            print("\nEmergency stop activated.")
        finally:
            self.stop()
            print("System shutdown.")

if __name__ == "__main__":
    car = TeamIoT_ThreeAngleAvoider()
    car.run()