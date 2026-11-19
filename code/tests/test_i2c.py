"""robot/i2c.py against a fake /dev/i2c (os.open / ioctl / read / write patched)."""
import unittest
from unittest import mock

from robot import i2c


class FakeBus:
    def __init__(self, reply=b"\x68", ioctl_error=False):
        self.reply, self.ioctl_error = reply, ioctl_error
        self.written, self.closed, self.ioctl_args = [], [], None

    def patches(self):
        return [mock.patch.object(i2c.os, "open", lambda path, flags: 7),
                mock.patch.object(i2c.os, "write", lambda fd, data: self.written.append(bytes(data)) or len(data)),
                mock.patch.object(i2c.os, "read", lambda fd, n: self.reply[:n]),
                mock.patch.object(i2c.os, "close", lambda fd: self.closed.append(fd)),
                mock.patch.object(i2c.fcntl, "ioctl", self._ioctl)]

    def _ioctl(self, fd, req, addr):
        if self.ioctl_error:
            raise OSError(16, "Device or resource busy")
        self.ioctl_args = (fd, req, addr)


class I2CDeviceTest(unittest.TestCase):
    def open(self, bus):
        for p in bus.patches():
            p.start()
            self.addCleanup(p.stop)
        return i2c.I2CDevice("/dev/i2c-1", 0x68)

    def test_sets_the_chip_address(self):
        bus = FakeBus()
        self.open(bus)
        self.assertEqual(bus.ioctl_args, (7, i2c.I2C_SLAVE, 0x68))

    def test_read_writes_the_register_then_reads(self):
        bus = FakeBus(reply=b"\x01\x02\x03")
        dev = self.open(bus)
        self.assertEqual(dev.read(0x3B, 3), b"\x01\x02\x03")
        self.assertEqual(bus.written, [b"\x3b"])

    def test_short_read_is_an_error(self):
        bus = FakeBus(reply=b"\x01")
        dev = self.open(bus)
        with self.assertRaisesRegex(OSError, "short read from 0x68: 1/14"):
            dev.read(0x3B, 14)

    def test_write_sends_register_and_value(self):
        bus = FakeBus()
        dev = self.open(bus)
        dev.write(0x6B, 0x180)                     # only the low byte goes out
        self.assertEqual(bus.written, [b"\x6b\x80"])

    def test_write_block_is_one_transaction(self):
        bus = FakeBus()
        dev = self.open(bus)
        dev.write_block(0x0A, [0, 0, 0x133, 0x01])  # PCA9685 channel: ON_L, ON_H, OFF_L, OFF_H
        self.assertEqual(bus.written, [b"\x0a\x00\x00\x33\x01"])

    def test_failed_address_set_closes_the_bus(self):
        bus = FakeBus(ioctl_error=True)
        with self.assertRaises(OSError):
            self.open(bus)
        self.assertEqual(bus.closed, [7])

    def test_close_twice_is_harmless(self):
        bus = FakeBus()
        dev = self.open(bus)
        dev.close()
        dev.close()
        self.assertEqual(bus.closed, [7])


if __name__ == "__main__":
    unittest.main()
