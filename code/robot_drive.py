#!/usr/bin/env python3
"""
Keyboard drive over SSH.
  w = forward   s = backward   a = left   d = right
  space = stop  + / - = speed  q = quit
Same wiring as robot_test.py.
"""
import sys, termios, tty
from gpiozero import Robot, Motor

robot = Robot(
    left=Motor(forward=27, backward=22, enable=17),
    right=Motor(forward=23, backward=24, enable=25),
)
speed = 0.5

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

print(__doc__)
try:
    while True:
        k = getch().lower()
        if k == 'w': robot.forward(speed)
        elif k == 's': robot.backward(speed)
        elif k == 'a': robot.left(speed)
        elif k == 'd': robot.right(speed)
        elif k == ' ': robot.stop()
        elif k == '+': speed = min(1.0, speed + 0.1)
        elif k == '-': speed = max(0.2, speed - 0.1)
        elif k in ('q', '\x03'): break
        print(f"\rkey={k!r} speed={speed:.1f}   ", end='', flush=True)
finally:
    robot.stop()
    print("\nstopped")
