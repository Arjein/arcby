#!/usr/bin/env python3
"""
Robot first drive test - Raspberry Pi 4 + L298N + 2 TT motors.

Wiring (Pi physical pin -> L298N). Same order as the L298N header,
left motor on the inner row, right motor on the outer row:
  pin 11 (GPIO17) -> ENA   (remove the ENA cap first)
  pin 13 (GPIO27) -> IN1
  pin 15 (GPIO22) -> IN2
  pin 16 (GPIO23) -> IN3
  pin 18 (GPIO24) -> IN4
  pin 22 (GPIO25) -> ENB   (remove the ENB cap first)
  pin 14 (GND)    -> L298N GND   <-- REQUIRED (common ground)
All motor pins are GPIO9 or higher: they boot LOW (pull-down), so the
motors stay still while the Pi boots. GPIO0-8 boot HIGH - never use them
for motor driver inputs.

Run:  python3 robot_test.py
Stop at any time: Ctrl+C (motors stop automatically).
"""
from time import sleep
from gpiozero import Robot, Motor

SPEED = 0.5   # 0.0 .. 1.0  - start slow

robot = Robot(
    left=Motor(forward=27, backward=22, enable=17),
    right=Motor(forward=23, backward=24, enable=25),
)

def step(name, action, seconds=1.0):
    print(f"-> {name}")
    action()
    sleep(seconds)
    robot.stop()
    sleep(0.7)

try:
    print("Single-wheel check: LEFT wheel forward only")
    robot.left_motor.forward(SPEED); sleep(1.0); robot.stop(); sleep(0.7)
    print("Single-wheel check: RIGHT wheel forward only")
    robot.right_motor.forward(SPEED); sleep(1.0); robot.stop(); sleep(0.7)

    step("forward",  lambda: robot.forward(SPEED))
    step("backward", lambda: robot.backward(SPEED))
    step("turn left (spin)",  lambda: robot.left(SPEED))
    step("turn right (spin)", lambda: robot.right(SPEED))
    print("Done.")
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    robot.stop()
