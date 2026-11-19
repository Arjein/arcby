"""
Minimal I2C register access through /dev/i2c-* (standard library only).
One I2CDevice per chip address; safe to share between threads.
"""
import fcntl
import os
import threading

I2C_SLAVE = 0x0703   # ioctl: set the chip address for this file descriptor


class I2CDevice:
    def __init__(self, bus, addr):
        self.addr = addr
        self._lock = threading.Lock()
        self._fd = os.open(bus, os.O_RDWR)
        try:
            fcntl.ioctl(self._fd, I2C_SLAVE, addr)
        except OSError:
            os.close(self._fd)
            raise

    def read(self, reg, n):
        """Read n bytes starting at register reg. Raises OSError if the chip does not answer."""
        with self._lock:
            os.write(self._fd, bytes([reg]))
            data = os.read(self._fd, n)
        if len(data) != n:
            raise OSError(f"short read from 0x{self.addr:02x}: {len(data)}/{n} bytes")
        return data

    def write(self, reg, value):
        with self._lock:
            os.write(self._fd, bytes([reg, value & 0xFF]))

    def write_block(self, reg, values):
        """Write several bytes from register reg on in ONE transaction (the chip must auto-increment)."""
        with self._lock:
            os.write(self._fd, bytes([reg, *(v & 0xFF for v in values)]))

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
