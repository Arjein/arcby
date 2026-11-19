"""hwtest/bringup.py: parsers, audio analysis and failure paths with a simulated robot."""
import argparse
import io
import math
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hwtest"))

import bringup                                    # noqa: E402
from common import FAIL, INFO, PASS, SKIP, Report  # noqa: E402
from robot import config, pan_tilt                # noqa: E402

PI_OS_CONFIG = """\
# For more options and information see
dtparam=audio=on
camera_auto_detect=1
[cm4]
dtparam=i2s=on
otg_mode=1
[all]
dtparam=i2c_arm=on,spi=off
dtoverlay=gpio-fan,gpiopin=4,temp=60000
gpio=17,22,23,24,25,27=ip,pd   # motor pins LOW at boot
#dtoverlay=googlevoicehat-soundcard
"""

APLAY_L = """\
**** List of PLAYBACK Hardware Devices ****
card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
  Subdevices: 8/8
card 1: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-codec-0 []
card 2: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
card 2: vc4hdmi0 [vc4-hdmi-0], device 1: second device []
"""


def wav_extensible_32(frames):
    """32-bit stereo WAV with a WAVE_FORMAT_EXTENSIBLE header, like arecord -f S32_LE writes."""
    pcm = b"".join(struct.pack("<ii", l, r) for l, r in frames)
    fmt = struct.pack("<HHIIHH", 0xFFFE, 2, 48000, 48000 * 8, 8, 32) + struct.pack("<HHI", 22, 32, 3) + bytes(16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def recorded(path):
    return Path(path).read_bytes()


class ParseTest(unittest.TestCase):
    def test_config_sections_and_comments(self):
        c = bringup.parse_config(PI_OS_CONFIG)
        self.assertIn("i2c_arm", c["dtparams"])
        self.assertNotIn("spi", c["dtparams"])            # =off does not count
        self.assertNotIn("i2s", c["dtparams"])            # only in [cm4]
        self.assertIn("audio", c["dtparams"])
        self.assertIn("gpio-fan", c["overlays"])
        self.assertNotIn("googlevoicehat-soundcard", c["overlays"])   # commented out
        self.assertIn("gpio=17,22,23,24,25,27=ip,pd", c["lines"])     # trailing comment stripped
        self.assertTrue(c["camera_auto_detect"])

    def test_cards(self):
        self.assertEqual([c[0] for c in bringup.parse_cards(APLAY_L)], [0, 1, 2])
        self.assertEqual(bringup.find_audio_card(APLAY_L), 1)
        self.assertIsNone(bringup.find_audio_card(APLAY_L.replace("googlevoi", "x").replace("voiceHAT", "x")
                                                  .replace("googlevoicehat", "x")))

    def test_cameras(self):
        listing = "Available cameras\n-----------------\n0 : imx708_wide [4608x2592 10-bit RGGB] (/base/...)\n"
        self.assertEqual(bringup.parse_cameras(listing), ["imx708_wide"])
        self.assertEqual(bringup.parse_cameras("No cameras available!\n"), [])

    def test_jpeg_ok(self):
        self.assertTrue(bringup.jpeg_ok(b"\xff\xd8" + bytes(6000) + b"\xff\xd9"))
        self.assertFalse(bringup.jpeg_ok(b"\xff\xd8" + bytes(6000)))           # cut off
        self.assertFalse(bringup.jpeg_ok(b"\xff\xd8\xff\xd9"))                 # too small


class WavTest(unittest.TestCase):
    def test_reads_extensible_32bit(self):
        ch, rate, bits, s = bringup.read_wav(wav_extensible_32([(1, -1), (2 ** 30, 0)]))
        self.assertEqual((ch, rate, bits, list(s)), (2, 48000, 32, [1, -1, 2 ** 30, 0]))

    def test_round_trip_16bit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.wav"
            bringup.write_wav16(p, [0.0, 0.5, -1.0])
            ch, rate, bits, s = bringup.read_wav(p.read_bytes())
        self.assertEqual((ch, bits), (2, 16))
        self.assertEqual(list(s), [0, 0, 16383, 16383, -32767, -32767])

    def test_rejects_non_wav(self):
        with self.assertRaises(ValueError):
            bringup.read_wav(b"hello world, not a wav")

    def test_beeps_length_and_level(self):
        b = bringup.beeps(rate=8000)
        self.assertEqual(len(b), 3 * (2000 + 800))
        self.assertLessEqual(max(abs(v) for v in b), bringup.SPEAKER_LEVEL)
        self.assertEqual(b[0], 0.0)                       # fades in


class MicTest(unittest.TestCase):
    def levels(self, loud):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.wav"
            bringup.fake_recording(p, loud)
            return bringup.channel_levels(p.read_bytes()), p.read_bytes()

    def test_voice_louder_than_quiet_passes_on_left(self):
        quiet, _ = self.levels(False)
        voice, _ = self.levels(True)
        result, ch, text = bringup.judge_mic(quiet, voice)
        self.assertEqual((result, ch), (PASS, 0))
        self.assertIn("LEFT", text)

    def test_no_difference_fails(self):
        quiet, _ = self.levels(False)
        result, ch, _ = bringup.judge_mic(quiet, quiet)
        self.assertEqual((result, ch), (FAIL, 0))

    def test_all_zeros_fails_without_channel(self):
        zeros = bringup.channel_levels(wav_extensible_32([(0, 0)] * 100))
        self.assertEqual(bringup.judge_mic(zeros, zeros), (FAIL, None, "mic sends only zeros"))

    def test_right_channel_detected(self):
        quiet = bringup.channel_levels(wav_extensible_32([(0, i % 3) for i in range(1000)]))
        voice = bringup.channel_levels(wav_extensible_32([(0, (i % 50 - 25) * 2 ** 22) for i in range(1000)]))
        result, ch, text = bringup.judge_mic(quiet, voice)
        self.assertEqual((result, ch), (PASS, 1))
        self.assertIn("RIGHT", text)

    def test_rumble_does_not_hide_the_voice(self):         # the real INMP441: -22 dBFS below 40 Hz
        quiet, _ = self.levels(False)
        voice, _ = self.levels(True)
        self.assertAlmostEqual(quiet[0]["rms_db"], voice[0]["rms_db"], delta=1.0)    # raw: the same
        self.assertGreater(voice[0]["speech_db"] - quiet[0]["speech_db"], 20)       # filtered: clear

    def test_highpass_keeps_speech_and_drops_rumble(self):
        rate = 8000
        rumble = [math.sin(2 * math.pi * 25 * i / rate) for i in range(rate)]
        tone = [math.sin(2 * math.pi * 1000 * i / rate) for i in range(rate)]
        rms = lambda v: math.sqrt(sum(x * x for x in v[rate // 4:]) / len(v[rate // 4:]))
        self.assertLess(rms(bringup.highpass(rumble, rate)), 0.01 * rms(rumble))
        self.assertGreater(rms(bringup.highpass(tone, rate)), 0.95 * rms(tone))

    def test_playback_is_normalised(self):
        _, data = self.levels(True)
        x = bringup.playback_samples(data, 0)
        self.assertAlmostEqual(max(abs(v) for v in x), bringup.SPEAKER_LEVEL, places=6)


class Broken(bringup.SimHw):
    """Simulated robot with selected faults."""

    def __init__(self, **faults):
        super().__init__()
        self.faults = faults

    def robot_processes(self):
        return ["1234 python3 drive_web.py"] if self.faults.get("robot_running") else []

    def i2c_scan(self):
        found = super().i2c_scan()
        if self.faults.get("no_ptz"):
            found.pop(config.PTZ_ADDR)
            found[0x41] = "ok"
        return found

    def throttled(self):
        return self.faults.get("throttled", 0)

    def run(self, cmd, timeout=10):
        if cmd[0] == "rpicam-still" and self.faults.get("camera_busy"):
            return 255, "", "ERROR: *** failed to acquire camera /base/soc/i2c0mux/i2c@1/imx708@1a ***\n"
        if cmd[0] in ("aplay", "arecord") and "-l" in cmd and self.faults.get("no_audio"):
            return 0, APLAY_L.split("card 1")[0], ""
        if cmd[0] == "arecord" and "-l" not in cmd and self.faults.get("dead_mic"):
            Path(cmd[-1]).write_bytes(wav_extensible_32([(0, 0)] * 1000))
            return 0, "", ""
        return super().run(cmd, timeout)

    def config_text(self):
        return PI_OS_CONFIG if self.faults.get("plain_config") else super().config_text()


class SessionTest(unittest.TestCase):
    def ctx(self, **faults):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        args = argparse.Namespace(yes=True, sim=True)
        return bringup.Ctx(args, Broken(**faults), Report("t"), Path(self.tmp.name))

    def run_test(self, ctx, *names):
        with redirect_stdout(io.StringIO()):
            for n in names:
                bringup.TESTS[n](ctx) if n in bringup.TESTS else getattr(bringup, n)(ctx)
        return {t: (r, d) for t, r, d in ctx.rep.rows}

    def test_running_robot_software_aborts(self):
        ctx = self.ctx(robot_running=True)
        with redirect_stdout(io.StringIO()):
            self.assertFalse(bringup.preflight(ctx))
        self.assertTrue(ctx.rep.failed)

    def test_missing_ptz_skips_pantilt_and_names_the_stranger(self):
        ctx = self.ctx(no_ptz=True)
        rows = self.run_test(ctx, "i2c", "pantilt")
        self.assertEqual(rows["I2C 0x40 PTZ board (PCA9685)"][0], FAIL)
        self.assertEqual(rows["Pan-tilt"][0], SKIP)
        self.assertTrue(any("0x41" in h for h in ctx.rep.hints))
        self.assertEqual(ctx.hw.ptz.writes, [])               # nothing written to the board

    def test_pantilt_leaves_head_centred(self):
        ctx = self.ctx()
        rows = self.run_test(ctx, "pantilt")
        self.assertEqual(rows["Look-around sweep"][0], PASS)
        prescale = pan_tilt.prescale_for(config.PTZ_PWM_HZ)
        for ch, centre in ((config.PTZ_PAN_CH, config.PTZ_PAN_CENTRE),
                           (config.PTZ_TILT_CH, config.PTZ_TILT_CENTRE)):
            self.assertEqual(ctx.hw.ptz.counts(ch), pan_tilt.us_to_counts(pan_tilt.angle_to_us(centre), prescale))

    def test_camera_busy_hint(self):
        ctx = self.ctx(camera_busy=True)
        rows = self.run_test(ctx, "camera")
        self.assertEqual(rows["Camera photo"][0], FAIL)
        self.assertTrue(any("dashboard" in h for h in ctx.rep.hints))

    def test_missing_sound_card(self):
        ctx = self.ctx(no_audio=True)
        rows = self.run_test(ctx, "speaker", "mic")
        self.assertEqual(rows["Speaker: sound card"][0], FAIL)
        self.assertEqual(rows["Mic: sound card"][0], FAIL)

    def test_dead_mic(self):
        ctx = self.ctx(dead_mic=True)
        rows = self.run_test(ctx, "mic")
        self.assertEqual(rows["Mic levels"], (FAIL, "mic sends only zeros"))
        self.assertNotIn("Mic -> speaker playback", rows)

    def test_config_reports_missing_lines(self):
        ctx = self.ctx(plain_config=True)
        rows = self.run_test(ctx, "config")
        self.assertEqual(rows["config: I2S on"][0], FAIL)
        self.assertEqual(rows["config: audio overlay"][0], FAIL)
        self.assertEqual(rows["config: onboard audio off"][0], INFO)
        self.assertEqual(rows["config: I2C bus on"][0], PASS)
        self.assertTrue(any("dtoverlay=googlevoicehat-soundcard" in h for h in ctx.rep.hints))

    def test_under_voltage_during_tests(self):
        ctx = self.ctx()
        self.run_test(ctx, "power")
        ctx.hw.faults["throttled"] = 0x50000
        rows = self.run_test(ctx, "test_power_after")
        self.assertEqual(rows["Pi power after tests"][0], FAIL)

    def test_old_under_voltage_is_only_info(self):
        ctx = self.ctx(throttled=0x50000)
        rows = self.run_test(ctx, "power", "test_power_after")
        self.assertEqual(rows["Pi power flags"][0], INFO)
        self.assertEqual(rows["Pi power after tests"][0], PASS)


if __name__ == "__main__":
    unittest.main()
