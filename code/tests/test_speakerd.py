"""robot/speakerd.py: the speaker's own process - protocol, timing, the ALSA set-up (with a fake libasound)."""
import ctypes
import io
import struct
import unittest

from robot import speakerd


def request(pid, data):
    return struct.pack("<4sII", b"PLAY", pid, len(data)) + data


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class ServeTest(unittest.TestCase):
    def test_each_phrase_is_reported_with_the_time_it_ends(self):
        clock = Clock()
        pcm = speakerd.NullPcm(clock=clock)
        inp = io.BytesIO(request(1, bytes(4 * 4800)) + request(2, bytes(4 * 9600)))
        out = io.StringIO()
        slept = []
        self.assertEqual(speakerd.serve(pcm, inp, out, clock=clock, sleep=slept.append), 0)
        lines = out.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("ready null S16_LE"))
        self.assertEqual(lines[1].split()[:2], ["done", "1"])
        self.assertAlmostEqual(float(lines[1].split()[2]), 100.1, places=3)     # 0.1 s of sound from now
        self.assertAlmostEqual(float(lines[2].split()[2]), 100.3, places=3)     # queued right after it
        self.assertAlmostEqual(slept[0], 0.3, places=3)          # at EOF: lets the last phrase finish

    def test_stop_fades_out_what_plays_and_drops_what_was_sent_before(self):     # barge-in
        import time as _t

        class SlowPcm(speakerd.NullPcm):
            def write(self, data):
                _t.sleep(0.005)                     # a real card blocks: the STOP arrives while phrase 1 plays
                return super().write(data)
        pcm = SlowPcm()
        one, two = bytes(4 * 48000), b"\x01\x00" * 2 * 4800       # 1 s, then 0.1 s after the STOP
        inp = io.BytesIO(request(1, one) + struct.pack("<4sII", b"STOP", 0, 0) + request(2, two))
        out = io.StringIO()
        self.assertEqual(speakerd.serve(pcm, inp, out, sleep=lambda s: None), 0)
        done = [ln.split()[1] for ln in out.getvalue().splitlines() if ln.startswith("done")]
        self.assertEqual(done, ["1", "2"])                         # both reported: nobody waits for ever
        self.assertLess(pcm.written, 48000 // 4 + 4800)            # phrase 1 cut short (+ its 10 ms fade)
        self.assertGreaterEqual(pcm.written, 4800)                 # phrase 2 played whole

    def test_fade_out_ends_silent(self):
        from array import array
        a = array("h", [10000] * 2 * 480)
        f = array("h")
        f.frombytes(speakerd.fade_out(a.tobytes()))
        self.assertEqual(len(f), 960)
        self.assertGreater(f[0], 9000)
        self.assertEqual(f[-1], 0)

    def test_a_bad_header_is_an_error(self):
        out = io.StringIO()
        code = speakerd.serve(speakerd.NullPcm(), io.BytesIO(b"JUNKJUNKJUNK"), out, sleep=lambda s: None)
        self.assertEqual(code, 2)
        self.assertIn("error bad header", out.getvalue())

    def test_a_cut_off_phrase_at_eof_just_ends(self):
        out = io.StringIO()
        code = speakerd.serve(speakerd.NullPcm(), io.BytesIO(request(1, bytes(400))[:20]), out,
                              sleep=lambda s: None)
        self.assertEqual(code, 0)


class FakeAsound:
    """Just enough libasound: answers the set-up, records sw params and writes, plays at a settable pace."""

    def __init__(self, formats=(speakerd.FORMAT_S32_LE,), avail=0):
        self.formats, self.avail, self.calls, self.written, self.sw = formats, avail, [], [], {}

    def __getattr__(self, name):
        def call(*args):
            self.calls.append(name)
            return 0
        return call

    def snd_strerror(self, err):
        return b"fake error"

    def snd_pcm_hw_params_test_format(self, pcm, hw, fmt):
        return 0 if fmt in self.formats else -22

    def snd_pcm_hw_params_get_buffer_size(self, hw, ref):
        ref._obj.value = 7200
        return 0

    def snd_pcm_hw_params_get_period_size(self, hw, ref, d):
        ref._obj.value = 1440
        return 0

    def snd_pcm_sw_params_get_boundary(self, sw, ref):
        ref._obj.value = 1 << 40
        return 0

    def _sw(name):
        def set_(self, pcm, sw, value):
            self.sw[name] = value.value if isinstance(value, ctypes.c_ulong) else value
            return 0
        return set_
    snd_pcm_sw_params_set_stop_threshold = _sw("stop")
    snd_pcm_sw_params_set_silence_threshold = _sw("silence_threshold")
    snd_pcm_sw_params_set_silence_size = _sw("silence_size")
    snd_pcm_sw_params_set_start_threshold = _sw("start")

    def snd_pcm_avail(self, pcm):
        return self.avail

    def snd_pcm_forward(self, pcm, n):
        self.calls.append(("forward", n.value))
        return n.value

    def snd_pcm_writei(self, pcm, buf, n):
        self.written.append(bytes(buf))
        return n.value


class AlsaPcmTest(unittest.TestCase):
    def test_the_card_never_stops_and_plays_silence_by_itself(self):
        lib = FakeAsound()
        pcm = speakerd.AlsaPcm("hw:0,0", lib=lib)
        self.assertEqual(lib.sw["stop"], 1 << 40)               # an empty buffer never stops the card
        self.assertEqual(lib.sw["silence_size"], 1 << 40)       # ALSA writes the silence itself
        self.assertEqual(lib.sw["silence_threshold"], 0)
        self.assertEqual((pcm.buffer_frames, pcm.period_frames, pcm.fmt), (7200, 1440, "S32_LE"))
        self.assertEqual(lib.written[0], bytes(1440 * 8))      # started at once with one period of silence

    def test_16_bit_frames_go_to_a_32_bit_card_in_the_top_bytes(self):
        lib = FakeAsound()
        pcm = speakerd.AlsaPcm("hw:0,0", lib=lib)
        lib.written.clear()
        pcm.write(struct.pack("<hh", 0x1234, -2))
        self.assertEqual(lib.written, [struct.pack("<ii", 0x1234 << 16, -2 << 16)])

    def test_after_silence_the_phrase_starts_a_period_ahead_of_the_playing_position(self):
        lib = FakeAsound(formats=(speakerd.FORMAT_S16_LE,))
        pcm = speakerd.AlsaPcm("hw:0,0", lib=lib)
        lib.avail = 7200 + 48000                # ALSA played a second of silence past our last frame
        pcm.write(bytes(400))
        self.assertIn(("forward", 48000 + 1440), lib.calls)
        lib.calls.clear()
        lib.avail = 3000                        # still playing the previous phrase: queue right after it
        pcm.write(bytes(400))
        self.assertFalse([c for c in lib.calls if isinstance(c, tuple)])


if __name__ == "__main__":
    unittest.main()
