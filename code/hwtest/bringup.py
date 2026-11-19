#!/usr/bin/env python3
"""
Bring-up check: every module on the assembled robot, one at a time, with a
PASS / FAIL per step and a hint for each failure. No motors (the L298N signal
jumpers are not connected yet) - the robot can stand on the table.

  python3 code/hwtest/bringup.py                    # everything (~3 min, a few questions)
  python3 code/hwtest/bringup.py --only camera,mic
  python3 code/hwtest/bringup.py --sim --yes        # no hardware: tests the script itself

Steps, in this order (names for --only):
  power     Pi power flags (vcgencmd get_throttled)
  config    /boot/firmware/config.txt has the lines for I2C, I2S audio, camera, fan, motor pins
  i2c       bus scan: MPU6050 (0x68) + PTZ board (0x40) answer, nothing unexpected
  imu       MPU6050 wakes up, reads 1 g while standing still
  distance  HC-SR04 answers (it is on 3.3 V - see distance.py)
  pantilt   head centres, turns left, looks up, sweeps smoothly (you watch and answer)
  camera    camera detected, takes a photo
  speaker   three rising beeps (you listen)
  mic       3 s quiet + 3 s talking, levels compared, your voice played back
At the end the power flags are read again (servos / amp can make the Pi sag).

Stop the robot software and close the dashboard's camera view first: both use the
same hardware. Photo, recordings and the report go to ~/scratch/hwtest/.
"""
import argparse
import fcntl
import math
import os
import re
import struct
import subprocess
import sys
import time
import wave
from array import array
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from common import FAIL, INFO, PASS, SKIP, Report, heading
from robot import config, imu
from robot.i2c import I2C_SLAVE, I2CDevice
from robot.pan_tilt import PanTilt, PanTiltError

CONFIG_TXT = "/boot/firmware/config.txt"
ROBOT_PROCS = r"drive_web\.py|drive\.py|robot\.main|hwtest/(base|route|motors|imu)\.py"
EXPECTED_I2C = {config.IMU_ADDR: "MPU6050 IMU", config.PTZ_ADDR: "PTZ board (PCA9685)"}
OPTIONAL_I2C = {0x70: "PTZ board all-call address (PCA9685 default, gone after init)"}
AUDIO_CARD = re.compile(r"googlevoi|voicehat", re.I)
RATE = 48000                # I2S sample rate (googlevoicehat-soundcard)
REC_S = 3
SPEAKER_LEVEL = 0.08         # peak of test sounds (full scale = 1). The card has no volume control;
                            # 0.3 was loud at night, a 1 kHz beep at 0.15 reached the mic ~16000x over quiet
MIC_MIN_GAIN_DB = 10.0      # talking must be at least this much louder than quiet (above MIC_HIGHPASS_HZ)
MIC_HIGHPASS_HZ = 150       # the INMP441's rumble out before comparing (robot/mic.py does the same at 100 Hz)
UNDER_VOLTAGE_NOW, UNDER_VOLTAGE_SINCE_BOOT = 0x1, 0x10000

# Lines bring-up expects in config.txt: (what, check, required, line to add)
CONFIG_CHECKS = [
    ("I2C bus on", lambda c: "i2c_arm" in c["dtparams"], True, "dtparam=i2c_arm=on"),
    ("I2S on", lambda c: "i2s" in c["dtparams"], True, "dtparam=i2s=on"),
    ("audio overlay", lambda c: "googlevoicehat-soundcard" in c["overlays"], True,
     "dtoverlay=googlevoicehat-soundcard"),
    ("onboard audio off", lambda c: "audio" not in c["dtparams"], False, "#dtparam=audio=on"),
    ("camera", lambda c: c["camera_auto_detect"] or "imx708" in c["overlays"], True, "camera_auto_detect=1"),
    ("fan control", lambda c: "gpio-fan" in c["overlays"], False, "dtoverlay=gpio-fan,gpiopin=4,temp=60000"),
    ("motor pins LOW at boot", lambda c: "gpio=17,22,23,24,25,27=ip,pd" in c["lines"], True,
     "gpio=17,22,23,24,25,27=ip,pd"),
]


# ---------- pure helpers (unit tested in tests/test_bringup.py) ----------

def parse_config(text):
    """Active settings of config.txt for a Pi 4: sections other than [all] / [pi4] are skipped
    (Pi OS has [cm4] / [cm5] blocks), comments ignored. dtparam values may be comma lists."""
    section, lines, dtparams, overlays = "all", set(), set(), set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section not in ("all", "pi4"):
            continue
        lines.add(line)
        key, _, value = line.partition("=")
        if key == "dtparam":
            for p in value.split(","):
                name, _, v = p.partition("=")
                if v.lower() not in ("off", "0", "false", "no"):
                    dtparams.add(name.strip())
        elif key == "dtoverlay" and value:
            overlays.add(value.split(",")[0].strip())
    return {"lines": lines, "dtparams": dtparams, "overlays": overlays,
            "camera_auto_detect": "camera_auto_detect=1" in lines}


def parse_cards(text):
    """`aplay -l` / `arecord -l` -> [(card number, id, name)], one per card."""
    cards = {}
    for m in re.finditer(r"^card (\d+): (\S+) \[([^\]]*)\]", text, re.M):
        cards.setdefault(int(m.group(1)), (int(m.group(1)), m.group(2), m.group(3)))
    return list(cards.values())


def find_audio_card(text):
    for num, ident, name in parse_cards(text):
        if AUDIO_CARD.search(ident) or AUDIO_CARD.search(name):
            return num
    return None


def parse_cameras(text):
    """`rpicam-hello --list-cameras` -> sensor names, e.g. ['imx708_wide']."""
    return re.findall(r"^\s*\d+\s*:\s*(\S+)", text, re.M)


def jpeg_ok(data, min_bytes=5000):
    return len(data) >= min_bytes and data[:2] == b"\xff\xd8" and data.rstrip(b"\x00")[-2:] == b"\xff\xd9"


def read_wav(data):
    """RIFF/WAVE bytes -> (channels, rate, bits, sample array). Plain chunk walk, so it also reads
    the WAVE_FORMAT_EXTENSIBLE header arecord writes for 32-bit (Python 3.11's wave cannot)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = channels, rate, bits
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("WAV without fmt or data chunk")
    channels, rate, bits = fmt
    typecode = {16: "h", 32: "i"}.get(bits)
    if typecode is None:
        raise ValueError(f"{bits}-bit WAV not supported")
    samples = array(typecode)
    samples.frombytes(pcm[:len(pcm) - len(pcm) % samples.itemsize])
    if sys.byteorder == "big":
        samples.byteswap()
    return channels, rate, bits, samples


def highpass(x, rate, hz=MIC_HIGHPASS_HZ):
    """4th-order high-pass (two 2nd-order Butterworth sections), DC removed first. The INMP441's signal is
    almost all rumble / drift below 40 Hz (2026-11-01: -22 dBFS of it, the room's speech band at -87 dBFS):
    without this, quiet and talking measure the same."""
    mean = sum(x) / (len(x) or 1)
    y = [v - mean for v in x]
    w = 2 * math.pi * hz / rate
    cos, alpha = math.cos(w), math.sin(w) / (2 * math.sqrt(0.5))
    a0 = 1 + alpha
    b0, b1, b2 = (1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0
    a1, a2 = -2 * cos / a0, (1 - alpha) / a0
    for _ in range(2):
        out, x1, x2, y1, y2 = [], 0.0, 0.0, 0.0, 0.0
        for v in y:
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, o
            out.append(o)
        y = out
    return y


def channel_levels(data):
    """Per channel: rms (dBFS, DC removed), speech-band rms (dBFS, after highpass(), the filter's first 0.1 s
    left out), peak (0-1 of full scale, speech band), share of exact zeros."""
    channels, rate, bits, samples = read_wav(data)
    full = float(1 << (bits - 1))
    out = []
    for c in range(channels):
        ch = samples[c::channels]
        n = len(ch) or 1
        mean = sum(ch) / n
        rms = math.sqrt(sum((x - mean) ** 2 for x in ch) / n) / full
        hp = highpass(ch, rate)
        hp = hp[min(rate // 10, len(hp) // 10):] or [0.0]
        speech = math.sqrt(sum(v * v for v in hp) / len(hp)) / full
        out.append({"rms_db": 20 * math.log10(rms) if rms > 0 else -200.0,
                    "speech_db": 20 * math.log10(speech) if speech > 0 else -200.0,
                    "peak": max(abs(v) for v in hp) / full, "zeros": ch.count(0) / n})
    return out


def judge_mic(quiet, voice):
    """(result, channel index, text) from the levels of the quiet and the talking recording."""
    if all(c["zeros"] > 0.99 for c in quiet + voice):
        return FAIL, None, "mic sends only zeros"
    ch = max(range(len(voice)), key=lambda i: voice[i]["speech_db"])
    gain = voice[ch]["speech_db"] - quiet[ch]["speech_db"]
    side = ("LEFT", "RIGHT")[ch] if len(voice) == 2 else str(ch)
    text = (f"{side} channel, above {MIC_HIGHPASS_HZ} Hz: quiet {quiet[ch]['speech_db']:.0f} dBFS, talking "
            f"{voice[ch]['speech_db']:.0f} dBFS ({gain:+.0f} dB, peak {voice[ch]['peak']:.1%}; rumble below: "
            f"{voice[ch]['rms_db']:.0f} dBFS)")
    return (PASS if gain >= MIC_MIN_GAIN_DB else FAIL), ch, text


def write_wav16(path, mono, rate=RATE):
    """Floats in -1..1 -> 16-bit stereo WAV (same sound on both channels)."""
    pcm = array("h")
    for v in mono:
        s = int(max(-1.0, min(1.0, v)) * 32767)
        pcm.extend((s, s))
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def beeps(freqs=(440, 660, 880), each_s=0.25, gap_s=0.1, level=SPEAKER_LEVEL, rate=RATE):
    """Three rising beeps with a 10 ms fade at each end (no clicks)."""
    out, fade = [], int(0.01 * rate)
    for f in freqs:
        n = int(each_s * rate)
        for i in range(n):
            env = min(1.0, i / fade, (n - 1 - i) / fade)
            out.append(level * env * math.sin(2 * math.pi * f * i / rate))
        out.extend([0.0] * int(gap_s * rate))
    return out


def playback_samples(data, ch, peak=SPEAKER_LEVEL):
    """One channel of a recording, the rumble filtered out and scaled so its peak is `peak` (the mic is quiet)."""
    channels, rate, _, samples = read_wav(data)
    x = highpass(samples[ch::channels], rate)
    x[:rate // 10] = [0.0] * len(x[:rate // 10])        # the filter settling
    top = max((abs(v) for v in x), default=0) or 1
    return [v / top * peak for v in x]


# ---------- hardware access (real and simulated) ----------

class RealHw:
    sim = False

    def run(self, cmd, timeout=10):
        """(exit code, stdout, stderr); 127 if the program is missing."""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except FileNotFoundError:
            return 127, "", f"{cmd[0]}: not installed"
        except subprocess.TimeoutExpired:
            return 124, "", f"{cmd[0]}: no result within {timeout} s"

    def throttled(self):
        code, out, err = self.run(["vcgencmd", "get_throttled"], 5)
        try:
            return int(out.strip().split("=")[1], 16)
        except (IndexError, ValueError):
            return None

    def config_text(self):
        return Path(CONFIG_TXT).read_text()

    def robot_processes(self):
        _, out, _ = self.run(["pgrep", "-af", ROBOT_PROCS], 5)
        return [l for l in out.splitlines() if l.strip()]

    def i2c_scan(self):
        """{addr: 'ok' | 'driver'} like `i2cdetect -r` (a 1-byte read per address, no writes)."""
        fd = os.open(config.I2C_BUS, os.O_RDWR)
        found = {}
        try:
            for addr in range(0x08, 0x78):
                try:
                    fcntl.ioctl(fd, I2C_SLAVE, addr)
                except OSError:
                    found[addr] = "driver"      # a kernel driver owns it (i2cdetect: UU)
                    continue
                try:
                    os.read(fd, 1)
                    found[addr] = "ok"
                except OSError:
                    pass
        finally:
            os.close(fd)
        return found

    def i2c(self, addr):
        return I2CDevice(config.I2C_BUS, addr)

    def distance_sensor(self):
        from distance import Sensor          # gpiozero, only on the Pi
        return Sensor()

    def sleep(self, s):
        time.sleep(s)


class SimHw(RealHw):
    """Healthy robot without hardware: fake chips and canned command output."""
    sim = True

    def __init__(self):
        from tests.fakes import FakeMpu6050, FakePca9685
        self.mpu, self.ptz = FakeMpu6050(), FakePca9685()
        self.records = 0

    def run(self, cmd, timeout=10):
        prog = cmd[0]
        if prog == "rpicam-hello":
            return 0, "Available cameras\n-----------------\n0 : imx708_wide [4608x2592 10-bit RGGB] (/base/soc/i2c0mux/i2c@1/imx708@1a)\n", ""
        if prog == "rpicam-still":
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xff\xd8" + bytes(8000) + b"\xff\xd9")
            return 0, "", ""
        if prog in ("aplay", "arecord") and "-l" in cmd:
            kind = "PLAYBACK" if prog == "aplay" else "CAPTURE"
            return 0, (f"**** List of {kind} Hardware Devices ****\n"
                       "card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: "
                       "Google voiceHAT SoundCard HiFi voicehat-codec-0 [Google voiceHAT SoundCard HiFi voicehat-codec-0]\n"), ""
        if prog == "arecord":
            self.records += 1
            fake_recording(cmd[-1], loud=self.records % 2 == 0)
            return 0, "", ""
        if prog == "aplay":
            return 0, "", ""
        return 127, "", f"{prog}: not simulated"

    def throttled(self):
        return 0

    def config_text(self):
        return "\n".join(line for *_, line in CONFIG_CHECKS if not line.startswith("#")) + "\ndtparam=audio=off\n"

    def robot_processes(self):
        return []

    def i2c_scan(self):
        return {config.PTZ_ADDR: "ok", config.IMU_ADDR: "ok", 0x70: "ok"}

    def i2c(self, addr):
        return {config.IMU_ADDR: self.mpu, config.PTZ_ADDR: self.ptz}[addr]

    def distance_sensor(self):
        class Fake:
            def pings(self, n):
                return [42.0 + 0.1 * i for i in range(n)]

            def close(self):
                pass
        return Fake()

    def sleep(self, s):
        pass


def fake_recording(path, loud):
    """32-bit stereo like the INMP441 on the left channel: a big 25 Hz rumble + quiet hiss, or + a 300 Hz 'voice'."""
    n, amp = RATE * REC_S, (0.005 if loud else 0.0001)     # like the real one: voice ~35 dB under the rumble
    pcm = array("i")
    for i in range(n):
        v = amp * math.sin(2 * math.pi * 300 * i / RATE) + 0.0002 * ((i * 7919) % 101 - 50) / 50 \
            + 0.08 * math.sin(2 * math.pi * 25 * i / RATE)
        pcm.extend((int(v * 2 ** 31), 0))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(4)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


# ---------- session ----------

class Ctx:
    def __init__(self, args, hw, rep, folder):
        self.args, self.hw, self.rep, self.folder = args, hw, rep, folder
        self.found = None           # I2C scan result, None = not scanned
        self.power_start = None

    def ask(self, prompt, choices="yn"):
        if self.args.yes:
            print(f"{prompt} [auto: {choices[0]}]")
            return choices[0]
        return common.getkey(prompt, choices)

    def present(self, addr):
        return self.found is None or addr in self.found


def power_text(value):
    if value is None:
        return "could not read"
    flags = []
    if value & UNDER_VOLTAGE_NOW:
        flags.append("UNDER-VOLTAGE NOW")
    if value & UNDER_VOLTAGE_SINCE_BOOT:
        flags.append("under-voltage since boot")
    return f"throttled=0x{value:x}" + (f" ({', '.join(flags)})" if flags else "")


def test_power(ctx):
    heading("Power")
    rep, value = ctx.rep, ctx.hw.throttled()
    ctx.power_start = value
    if value is None:
        rep.add("Pi power flags", SKIP, "vcgencmd not available")
    elif value & UNDER_VOLTAGE_NOW:
        rep.add("Pi power flags", FAIL, power_text(value))
        rep.hint("Under-voltage right now: the battery is low (LiPo alarm < 3.5 V/cell = charge it) or the "
                 "Pi buck is below 5.1 V. Fix this first - every other test can fail from it.")
    elif value & UNDER_VOLTAGE_SINCE_BOOT:
        rep.add("Pi power flags", INFO, power_text(value) + " - earlier, not now")
    else:
        rep.add("Pi power flags", PASS, power_text(value))


def test_power_after(ctx):
    heading("Power after the tests")
    rep, value = ctx.rep, ctx.hw.throttled()
    if value is None:
        rep.add("Pi power after tests", SKIP, "vcgencmd not available")
        return
    new = value & ~(ctx.power_start or 0)
    bad = value & UNDER_VOLTAGE_NOW or new & UNDER_VOLTAGE_SINCE_BOOT
    rep.add("Pi power after tests", FAIL if bad else PASS, power_text(value))
    if bad:
        rep.hint("The Pi sagged during the tests: servos and speaker draw from the same battery. "
                 "Check the battery voltage; if it is full, check the Pi buck output under load.")


def test_config(ctx):
    heading("config.txt")
    rep = ctx.rep
    try:
        cfg = parse_config(ctx.hw.config_text())
    except OSError as e:
        rep.add("config.txt", FAIL, str(e))
        return
    missing = []
    for what, check, required, line in CONFIG_CHECKS:
        if check(cfg):
            rep.add(f"config: {what}", PASS)
        else:
            rep.add(f"config: {what}", FAIL if required else INFO, f"missing: {line}")
            missing.append(line)
    if missing:
        rep.hint("Add to /boot/firmware/config.txt under [all] (`sudo nano /boot/firmware/config.txt`), "
                 "then `sudo reboot`: " + " | ".join(missing))


def test_i2c(ctx):
    heading("I2C bus")
    rep = ctx.rep
    try:
        found = ctx.hw.i2c_scan()
    except OSError as e:
        rep.add("I2C bus", FAIL, f"{config.I2C_BUS}: {e}")
        rep.hint("No I2C bus: `dtparam=i2c_arm=on` in config.txt (or raspi-config -> Interface -> I2C), reboot.")
        ctx.found = {}
        return
    ctx.found = found
    rep.fact("I2C devices: " + (", ".join(f"0x{a:02x}" + (" (driver)" if s == "driver" else "")
                                          for a, s in sorted(found.items())) or "none"))
    for addr, name in EXPECTED_I2C.items():
        rep.add(f"I2C 0x{addr:02x} {name}", PASS if addr in found else FAIL)
    unknown = [a for a in found if a not in EXPECTED_I2C and a not in OPTIONAL_I2C]
    if unknown:
        rep.add("I2C unexpected devices", INFO, ", ".join(f"0x{a:02x}" for a in sorted(unknown)))
    if not found:
        rep.hint("Nothing on the bus: SDA = Pi pin 3 -> breadboard A2 LEFT, SCL = pin 5 -> A3 LEFT; "
                 "MPU VCC on 3.3 V (A4), common GND.")
    elif config.IMU_ADDR not in found:
        rep.hint("MPU6050 missing: check its VCC (3.3 V, breadboard A4), GND (pin 25), SDA/SCL jumpers.")
    if found and config.PTZ_ADDR not in found:
        if unknown:
            rep.hint(f"PTZ board not at 0x{config.PTZ_ADDR:02x} but something answers at "
                     f"{', '.join(f'0x{a:02x}' for a in unknown)} - maybe that is the PTZ board: tell Claude "
                     "(PTZ_ADDR in robot/config.py).")
        else:
            rep.hint("PTZ board missing: its VCC/GND come from the 4.7 V rail (breadboard A1/A2 RIGHT) - is the "
                     "servo buck powered? SDA/SCL to A2/A3 LEFT; pin order VCC SDA SCL GND.")


def test_imu(ctx):
    heading("IMU (MPU6050)")
    rep = ctx.rep
    if not ctx.present(config.IMU_ADDR):
        rep.add("IMU reads", SKIP, "not on the I2C bus")
        return
    try:
        mpu = imu.Mpu6050(ctx.hw.i2c(config.IMU_ADDR), sleep=ctx.hw.sleep)
        who = mpu.init()
        samples = []
        for _ in range(50):
            samples.append(imu.to_units(imu.decode(mpu.read_raw())))
            ctx.hw.sleep(0.01)
    except (OSError, imu.ImuError) as e:
        rep.add("IMU reads", FAIL, str(e))
        rep.hint("IMU answered the scan but not the setup: loose SDA/SCL or VCC jumper.")
        return
    g = sum(math.sqrt(sum(v * v for v in a)) for a, _, _ in samples) / len(samples)
    gz = [gy[2] for _, gy, _ in samples]
    mean = sum(gz) / len(gz)
    noise = math.sqrt(sum((v - mean) ** 2 for v in gz) / len(gz))
    temp = samples[-1][2]
    ok = 0.9 <= g <= 1.1 and noise < 2.0 and -10 < temp < 80
    rep.add("IMU reads", PASS if ok else FAIL,
            f"WHO_AM_I 0x{who:02x}, |a| {g:.2f} g, gyro noise {noise:.2f} deg/s, {temp:.0f} C")
    if not ok:
        rep.hint("IMU values off: robot must stand still; |a| should be ~1.00 g. Full check: "
                 "`python3 code/hwtest/imu.py`.")


def test_distance(ctx):
    heading("Distance (HC-SR04)")
    rep = ctx.rep
    print("Point the robot at something 20-150 cm away (a wall, a box).")
    try:
        sensor = ctx.hw.distance_sensor()
    except Exception as e:      # noqa: BLE001 - gpiozero raises many kinds
        rep.add("Distance sensor", FAIL, f"pins GPIO5/6: {e}")
        rep.hint("GPIO5/6 busy: close the dashboard page (it pings the sensor while open) and stop other scripts.")
        return
    try:
        res = sensor.pings(10)
    finally:
        sensor.close()
        if not ctx.hw.sim:
            common.release_pins([5, 6])
    vals = sorted(r for r in res if isinstance(r, float))
    if len(vals) >= 8:
        rep.add("Distance sensor", PASS, f"median {vals[len(vals) // 2]:.1f} cm ({len(vals)}/10 readings)")
    else:
        rep.add("Distance sensor", FAIL, f"{len(vals)}/10 readings ({res.count('noanswer')} no answer)")
        rep.hint("HC-SR04: VCC 3.3 V (breadboard A4), GND pin 30, TRIG pin 29, ECHO pin 31. "
                 "Detailed test: `python3 code/hwtest/distance.py`.")


PAN_DEG, TILT_DEG = 40.0, 25.0
TURNS = {"l": "left", "r": "right", "u": "up", "d": "down", "n": "did not move"}


def ask_turn(ctx, expected, other_axis):
    """Ask which way the head moved; expected key first so --yes answers it."""
    keys = expected + "".join(k for k in "lrud" if k != expected) + "n"
    print("Which way did the camera turn?  " + "  ".join(f"{k} = {TURNS[k]}" for k in keys))
    k = ctx.ask("Direction?", keys)
    if k == expected:
        return PASS, TURNS[k]
    if k == "n":
        return FAIL, "did not move"
    if k in other_axis:
        return FAIL, f"moved {TURNS[k]}: pan and tilt channels are swapped"
    return FAIL, f"moved {TURNS[k]}: direction reversed"


def test_pantilt(ctx):
    heading("Pan-tilt head")
    rep = ctx.rep
    if not ctx.present(config.PTZ_ADDR):
        rep.add("Pan-tilt", SKIP, f"PTZ board not on the I2C bus at 0x{config.PTZ_ADDR:02x}")
        return
    print(f"The head will move: pan +-{PAN_DEG:.0f} deg, tilt +-{TILT_DEG:.0f} deg. "
          "Keep fingers and the camera ribbon clear.")
    if ctx.ask("Ready?") != "y":
        rep.add("Pan-tilt", SKIP, "not started")
        return
    head = PanTilt(ctx.hw.i2c(config.PTZ_ADDR), sleep=ctx.hw.sleep)
    try:
        head.init()
        rep.add("PTZ board setup", PASS, f"PCA9685 at 0x{config.PTZ_ADDR:02x}, {config.PTZ_PWM_HZ} Hz")
        head.set(0, 0)
        ctx.hw.sleep(0.8)
        centred = ctx.ask("Is the camera looking straight ahead and level?")
        rep.add("Head centre", PASS if centred == "y" else INFO,
                "straight + level" if centred == "y" else "not straight - centre needs tuning")
        if centred != "y":
            rep.hint("Head not straight at centre: tell Claude roughly how many degrees off (left/right, up/down) "
                     "-> PTZ_PAN_CENTRE / PTZ_TILT_CENTRE in robot/config.py.")

        head.move(PAN_DEG, 0)
        result, text = ask_turn(ctx, "l", "ud")
        rep.add(f"Pan +{PAN_DEG:.0f} (expect left)", result, text)
        head.centre()
        head.move(0, TILT_DEG)
        t_result, t_text = ask_turn(ctx, "u", "lr")
        rep.add(f"Tilt +{TILT_DEG:.0f} (expect up)", t_result, t_text)
        head.centre()
        for txt, sign_name in ((text, "PTZ_PAN_SIGN"), (t_text, "PTZ_TILT_SIGN")):
            if "reversed" in txt:
                rep.hint(f"Direction reversed: set {sign_name} = -1 in robot/config.py.")
            elif "swapped" in txt:
                rep.hint("Pan and tilt swapped: swap PTZ_PAN_CH and PTZ_TILT_CH in robot/config.py "
                         "(or swap the two servo plugs on the PTZ board).")
            elif "did not move" in txt:
                rep.hint("Servo did not move: is the 4.7 V rail on (servo buck, breadboard A1/A2 RIGHT)? "
                         "Servo plugs seated on the PTZ board?")

        print("Now a slow look-around.")
        for p, t in ((-PAN_DEG, 0), (-PAN_DEG, TILT_DEG), (PAN_DEG, TILT_DEG), (PAN_DEG, -15), (0, -15), (0, 0)):
            head.move(p, t, speed_dps=60)
        smooth = ctx.ask("Smooth, without buzzing, jitter or the Pi resetting?")
        rep.add("Look-around sweep", PASS if smooth == "y" else FAIL)
        if smooth != "y":
            rep.hint("Jittery servos: check the 4.7 V rail under load with the meter (should stay > 4.5 V) and "
                     "the PTZ GND. Buzzing only at one end = the servo hits its end stop: narrow the limits.")
        rep.fact("Head left centred and holding (the PTZ board keeps the servos powered).")
    except (OSError, PanTiltError) as e:
        rep.add("Pan-tilt", FAIL, str(e))
        rep.hint("PTZ board stopped answering: loose SDA/SCL, or the 4.7 V rail dropped (servo stall).")


def test_camera(ctx):
    heading("Camera")
    rep, hw = ctx.rep, ctx.hw
    code, out, err = hw.run(["rpicam-hello", "--list-cameras"], 15)
    names = parse_cameras(out + err)
    if not names:
        rep.add("Camera detected", FAIL, (err.strip().splitlines() or ["no cameras"])[-1])
        rep.hint("No camera: ribbon fully in both connectors (blue side toward the Ethernet/USB on the Pi), "
                 "latch pushed down; `camera_auto_detect=1` in config.txt; reboot after re-seating.")
        return
    rep.add("Camera detected", PASS, ", ".join(names))
    photo = ctx.folder / "photo.jpg"
    code, out, err = hw.run(["rpicam-still", "-n", "-t", "1500", "--width", "1280", "--height", "960",
                             "-o", str(photo)], 30)
    data = photo.read_bytes() if photo.exists() else b""
    if code == 0 and jpeg_ok(data):
        rep.add("Camera photo", PASS, f"{len(data) // 1024} KiB -> {photo}")
    else:
        last = (err.strip().splitlines() or [f"exit code {code}"])[-1]
        rep.add("Camera photo", FAIL, last)
        if re.search(r"busy|acquire", err, re.I):
            rep.hint("Camera busy: stop the dashboard's camera view (or any other camera program) and re-run "
                     "`bringup.py --only camera`.")
        else:
            rep.hint("Photo failed although the camera is detected: re-seat the ribbon at the camera end.")
        return
    print("Optional: open the dashboard (http://raspberrypi.local:8080/) -> Camera -> Start, and look at the picture.")
    k = ctx.ask("Is the picture upright and sharp? (s = skip)", "yns")
    if k == "n":
        rep.add("Camera picture", INFO, "not upright / not sharp")
        rep.hint("Picture upside down: use the dashboard's 'rotate' button and tell Claude (camera rotation setting). "
                 "Blurry: the Module 3 focuses automatically - check the protective film is off the lens.")
    elif k == "y":
        rep.add("Camera picture", PASS, "upright and sharp")


def audio_card(ctx, prog, name):
    code, out, err = ctx.hw.run([prog, "-l"], 5)
    card = find_audio_card(out)
    if card is None:
        cards = ", ".join(c[1] for c in parse_cards(out)) or "none"
        ctx.rep.add(f"{name}: sound card", FAIL, f"googlevoicehat card missing (cards: {cards})")
        ctx.rep.hint("No I2S sound card: add `dtparam=i2s=on` and `dtoverlay=googlevoicehat-soundcard` to "
                     "config.txt (comment out `dtparam=audio=on`), then `sudo reboot`.")
    return card


def test_speaker(ctx):
    heading("Speaker (MAX98357A amp)")
    rep, hw = ctx.rep, ctx.hw
    card = audio_card(ctx, "aplay", "Speaker")
    if card is None:
        return
    path = ctx.folder / "beeps.wav"
    write_wav16(path, beeps())
    print("Listen: three rising beeps.")
    code, _, err = hw.run(["aplay", "-q", "-D", f"plughw:{card},0", str(path)], 15)
    if code != 0:
        rep.add("Speaker plays", FAIL, (err.strip().splitlines() or [f"exit code {code}"])[-1])
        return
    heard = ctx.ask("Did you hear three rising beeps?")
    rep.add("Speaker plays", PASS if heard == "y" else FAIL, f"card {card}")
    if heard != "y":
        rep.hint("Silent speaker: amp VIN = Pi pin 2 (5 V, breadboard A6), GND A7, DIN = pin 40 (A10), "
                 "BCLK = pin 12 (A11), LRC = pin 35 (A12); both speaker wires tight in the amp's screw terminal.")


def record(ctx, card, name):
    path = ctx.folder / f"{name}.wav"
    code, _, err = ctx.hw.run(["arecord", "-q", "-D", f"plughw:{card},0", "-c", "2", "-r", str(RATE),
                               "-f", "S32_LE", "-d", str(REC_S), str(path)], REC_S + 10)
    if code != 0 or not path.exists():
        ctx.rep.add(f"Mic records ({name})", FAIL, (err.strip().splitlines() or [f"exit code {code}"])[-1])
        return None
    return path.read_bytes()


def test_mic(ctx):
    heading("Microphone (INMP441)")
    rep, hw = ctx.rep, ctx.hw
    card = audio_card(ctx, "arecord", "Mic")
    if card is None:
        return
    print(f"Step 1: stay QUIET for {REC_S} s.")
    ctx.ask("Ready?", "y")
    quiet = record(ctx, card, "quiet")
    if quiet is None:
        return
    print(f"Step 2: TALK to the robot for {REC_S} s from ~30 cm ('Hello Arcby, one two three...').")
    ctx.ask("Ready?", "y")
    print("   recording...")
    voice = record(ctx, card, "voice")
    if voice is None:
        return
    try:
        result, ch, text = judge_mic(channel_levels(quiet), channel_levels(voice))
    except ValueError as e:
        rep.add("Mic levels", FAIL, f"recording unreadable: {e}")
        return
    rep.add("Mic levels", result, text)
    if ch is None:
        rep.hint("Mic sends nothing: SD = Pi pin 38 (direct), SCK = pin 12 (breadboard A11), WS = pin 35 (A12), "
                 "VDD = pin 17 (3.3 V), GND = pin 39, L/R tied to GND.")
        return
    if result == FAIL:
        rep.hint("Mic hears no difference: the sound port in the deck must be over the mic's hole (bottom port, "
                 "hole facing UP); foam tape must not cover it.")
    elif ch == 1:
        rep.fact("Mic is on the RIGHT channel - L/R is probably not at GND (the software expects LEFT).")
    path = ctx.folder / "voice-playback.wav"
    write_wav16(path, playback_samples(voice, ch))
    print("Playing your recording back on the speaker...")
    hw.run(["aplay", "-q", "-D", f"plughw:{card},0", str(path)], REC_S + 10)
    heard = ctx.ask("Did you hear your own voice?")
    rep.add("Mic -> speaker playback", PASS if heard == "y" else FAIL)
    if heard != "y" and result == PASS:
        rep.hint("Levels were fine but the playback was not heard: the speaker test above decides - if beeps "
                 "worked, re-run `bringup.py --only mic` and talk louder / closer.")


TESTS = {"power": test_power, "config": test_config, "i2c": test_i2c, "imu": test_imu,
         "distance": test_distance, "pantilt": test_pantilt, "camera": test_camera,
         "speaker": test_speaker, "mic": test_mic}


def preflight(ctx):
    procs = ctx.hw.robot_processes()
    if procs:
        ctx.rep.add("Robot software stopped", FAIL, "; ".join(procs))
        ctx.rep.hint("Stop the robot software first (Ctrl+C in its terminal): it uses the same I2C chips, "
                     "and this check resets the IMU.")
        return False
    print("Robot on the table, standing still. Motors are not used.")
    return ctx.ask("Ready?") == "y"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated: " + ",".join(TESTS))
    ap.add_argument("--sim", action="store_true", help="no hardware: fake chips and commands")
    ap.add_argument("--yes", action="store_true", help="answer every question with the expected answer (only with --sim)")
    args = ap.parse_args()
    if args.yes and not args.sim:
        ap.error("--yes is only allowed with --sim (on the robot you must watch and listen)")
    names = args.only.split(",") if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = common.LOG_DIR / f"bringup-{stamp}{'-sim' if args.sim else ''}"
    folder.mkdir(parents=True, exist_ok=True)
    rep = Report("bringup-sim" if args.sim else "bringup")
    ctx = Ctx(args, SimHw() if args.sim else RealHw(), rep, folder)
    try:
        if preflight(ctx):
            for n in names:
                TESTS[n](ctx)
            if "power" in names or len(names) > 1:
                test_power_after(ctx)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        rep.add("Run", SKIP, "aborted with Ctrl+C")

    rep.fact(f"Files (photo, recordings): {folder}")
    print("\n" + rep.text())
    print(f"\nSaved: {rep.save()}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
