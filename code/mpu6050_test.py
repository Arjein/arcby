#!/usr/bin/env python3
"""
MPU6050 (GY-521) quick test - standard library only, reads /dev/i2c-1 directly.

Wiring for this bench test (Pi physical pins):
  VCC -> pin 1  (3.3V)   (final wiring too - see docs/pinout.md)
  GND -> pin 9  (GND)     final wiring: pin 25
  SDA -> pin 3  (GPIO2)
  SCL -> pin 5  (GPIO3)
  XDA, XCL, AD0, INT: not connected (AD0 floating/low -> address 0x68)

Run:  python3 mpu6050_test.py          (Ctrl+C to stop)
Tip:  keep the board still for the first second (gyro offset calibration).

On Arcby the board is mounted upside down; this test does not flip axes,
so lying flat chip-up you should see az ~ +1 g, chip-down az ~ -1 g.
"""
import fcntl
import math
import os
import struct
import sys
import time

BUS = "/dev/i2c-1"
ADDR = 0x68
I2C_SLAVE = 0x0703

PWR_MGMT_1 = 0x6B
WHO_AM_I = 0x75
ACCEL_XOUT_H = 0x3B
ACCEL_SCALE = 16384.0   # LSB per g at +-2 g (default)
GYRO_SCALE = 131.0      # LSB per deg/s at +-250 deg/s (default)


def open_bus():
    fd = os.open(BUS, os.O_RDWR)
    fcntl.ioctl(fd, I2C_SLAVE, ADDR)
    return fd


def write_reg(fd, reg, value):
    os.write(fd, bytes([reg, value]))


def read_regs(fd, reg, n):
    os.write(fd, bytes([reg]))
    return os.read(fd, n)


def read_all(fd):
    data = read_regs(fd, ACCEL_XOUT_H, 14)
    ax, ay, az, t, gx, gy, gz = struct.unpack(">hhhhhhh", data)
    return (ax / ACCEL_SCALE, ay / ACCEL_SCALE, az / ACCEL_SCALE,
            t / 340.0 + 36.53,
            gx / GYRO_SCALE, gy / GYRO_SCALE, gz / GYRO_SCALE)


def main():
    try:
        fd = open_bus()
    except FileNotFoundError:
        sys.exit(f"{BUS} not found - enable I2C: sudo raspi-config -> Interface Options -> I2C")
    try:
        who = read_regs(fd, WHO_AM_I, 1)[0]
    except OSError as e:
        sys.exit(f"No answer at 0x{ADDR:02x} ({e}). Check VCC/GND/SDA/SCL wiring.")
    print(f"WHO_AM_I = 0x{who:02x} (expected 0x68{'' if who == 0x68 else ' - clone chips may report 0x70/0x72/0x98, usually fine'})")

    write_reg(fd, PWR_MGMT_1, 0x00)   # wake up (chip starts in sleep mode)
    time.sleep(0.1)

    print("Calibrating gyro - keep the board still...")
    n, bias = 100, [0.0, 0.0, 0.0]
    for _ in range(n):
        g = read_all(fd)[4:]
        bias = [b + v / n for b, v in zip(bias, g)]
        time.sleep(0.005)

    yaw = 0.0
    last = time.monotonic()
    print("ax/ay/az in g, gx/gy/gz in deg/s, tilt from gravity, yaw = integrated gz")
    try:
        while True:
            ax, ay, az, temp, gx, gy, gz = read_all(fd)
            gx, gy, gz = gx - bias[0], gy - bias[1], gz - bias[2]
            now = time.monotonic()
            yaw += gz * (now - last)
            last = now
            roll = math.degrees(math.atan2(ay, az))
            pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
            print(f"\ra {ax:+.2f} {ay:+.2f} {az:+.2f} g | g {gx:+7.1f} {gy:+7.1f} {gz:+7.1f} dps | "
                  f"roll {roll:+6.1f} pitch {pitch:+6.1f} yaw {yaw:+7.1f} deg | {temp:4.1f} C   ",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
