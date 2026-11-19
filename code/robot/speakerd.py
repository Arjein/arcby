#!/usr/bin/env python3
"""
speakerd: the robot's speaker as its own small process, so nothing in alive.py can ever make the speaker pop.

Why: the voiceHAT driver switches the amplifier on when the sound card starts and OFF when it stops - every
switch is an audible "pffzzrt" (Selim, 2026-11-01). The card used to stop whenever its buffer ran dry (an ALSA
underrun), e.g. while alive.py was busy loading models. Here ALSA is set up so that it NEVER stops:
stop_threshold = boundary (an empty buffer is not an error) and silence_size = boundary with
silence_threshold = 0 (ALSA itself writes silence behind the playing position). The card starts once, at
start-up (one amplifier switch), and runs until this process exits. Nobody has to feed it silence.

Protocol (the client is robot/tts.py Speaker):
    stdin   a header struct "<4sII" = b"PLAY", phrase id, number of bytes; then that many bytes of
            48 kHz 16-bit little-endian stereo frames. EOF = finish what is queued and exit.
    stdout  one line each:
            "ready <device> <format> buffer=<frames> period=<frames>"   once, when the card is open and running
            "done <id> <t>"   the phrase is written; its last sample leaves the speaker at time.monotonic() == t
            "error <message>" something failed (the process exits after a fatal one)
            b"STOP" header (id 0, 0 bytes): someone talks over the robot - the phrase playing fades out (10 ms), every
            phrase sent before it is dropped (still reported "done"), the card keeps running (no pop)

    python3 code/robot/speakerd.py --device hw:0,0         # what tts.Speaker starts (plughw:0,0 if hw fails)
    python3 code/robot/speakerd.py --device null           # no card: real-time timing only (tests, the Mac)
    python3 code/robot/speakerd.py --device hw:0,0 --beep  # check on the robot: one quiet beep, then exit

Stdlib only (ctypes to libasound.so.2): it starts in a fraction of a second and needs no numpy.
"""
import argparse
import ctypes
import ctypes.util
import logging
import math
import struct
import queue
import sys
import threading
import time
from array import array

log = logging.getLogger("speakerd")

RATE = 48000
CHANNELS = 2
HEADER = struct.Struct("<4sII")
MAGIC = b"PLAY"
STOP = b"STOP"              # header with id 0, 0 bytes: stop what plays now, drop what was sent before (barge-in)
PIECE_FRAMES = 2400         # written 50 ms at a time: a STOP is noticed between pieces
FADE_FRAMES = 480           # a stopped phrase fades out over 10 ms
BUFFER_US = 150000          # ALSA ring: 150 ms. No underrun can stop the card, so it can be short (= little delay)
PERIOD_US = 30000
WRITE_FRAMES = 4800         # frames per write call (0.1 s)

# alsa/pcm.h
STREAM_PLAYBACK = 0
ACCESS_RW_INTERLEAVED = 3
FORMAT_S16_LE = 2
FORMAT_S32_LE = 10
EPIPE, ESTRPIPE = 32, 86


class AlsaError(RuntimeError):
    pass


class NullPcm:
    """No card: plays in real time on paper (delay = what would still be playing). For tests and the Mac."""

    name, fmt = "null", "S16_LE"

    def __init__(self, clock=time.monotonic, buffer_frames=int(RATE * BUFFER_US / 1e6),
                 period_frames=int(RATE * PERIOD_US / 1e6)):
        self.clock = clock
        self.buffer_frames, self.period_frames = buffer_frames, period_frames
        self.end_t = clock()
        self.written = 0

    def write(self, data):
        frames = len(data) // 4
        self.end_t = max(self.end_t, self.clock()) + frames / RATE
        self.written += frames
        return frames

    def delay(self):
        return max(0, int((self.end_t - self.clock()) * RATE))

    def close(self):
        pass


class AlsaPcm:
    """One ALSA playback PCM that never stops (see the module text). write() takes 16-bit stereo frames."""

    def __init__(self, device="hw:0,0", lib=None):
        self.lib = lib or self._load()
        self.name = device
        self.pcm = ctypes.c_void_p()
        self._check(self.lib.snd_pcm_open(ctypes.byref(self.pcm), device.encode(), STREAM_PLAYBACK, 0),
                    f"open {device}")
        try:
            self._setup()
        except Exception:
            self.lib.snd_pcm_close(self.pcm)
            raise

    @staticmethod
    def _load():
        path = ctypes.util.find_library("asound") or "libasound.so.2"
        try:
            lib = ctypes.CDLL(path)
        except OSError as e:
            raise AlsaError(f"no libasound ({e})") from None
        lib.snd_strerror.restype = ctypes.c_char_p
        for name in ("snd_pcm_writei", "snd_pcm_avail", "snd_pcm_forward"):
            getattr(lib, name).restype = ctypes.c_long
        return lib

    def _check(self, err, what):
        if err < 0:
            raise AlsaError(f"{what}: {self.lib.snd_strerror(err).decode(errors='replace')}")
        return err

    def _setup(self):
        lib, pcm = self.lib, self.pcm
        hw = ctypes.c_void_p()
        self._check(lib.snd_pcm_hw_params_malloc(ctypes.byref(hw)), "hw params")
        try:
            self._check(lib.snd_pcm_hw_params_any(pcm, hw), "hw params")
            self._check(lib.snd_pcm_hw_params_set_access(pcm, hw, ACCESS_RW_INTERLEAVED), "access")
            # the voiceHAT card takes 32-bit samples only; plughw would convert, hw needs us to
            for fmt, name in ((FORMAT_S16_LE, "S16_LE"), (FORMAT_S32_LE, "S32_LE")):
                if lib.snd_pcm_hw_params_test_format(pcm, hw, fmt) == 0:
                    self._check(lib.snd_pcm_hw_params_set_format(pcm, hw, fmt), "format")
                    self.fmt = name
                    break
            else:
                raise AlsaError(f"{self.name} takes neither S16_LE nor S32_LE")
            self._check(lib.snd_pcm_hw_params_set_channels(pcm, hw, CHANNELS), "channels")
            self._check(lib.snd_pcm_hw_params_set_rate(pcm, hw, RATE, 0), f"rate {RATE}")
            us = ctypes.c_uint(BUFFER_US)
            self._check(lib.snd_pcm_hw_params_set_buffer_time_near(pcm, hw, ctypes.byref(us), None), "buffer")
            us = ctypes.c_uint(PERIOD_US)
            self._check(lib.snd_pcm_hw_params_set_period_time_near(pcm, hw, ctypes.byref(us), None), "period")
            self._check(lib.snd_pcm_hw_params(pcm, hw), "hw params")
            size = ctypes.c_ulong()
            self._check(lib.snd_pcm_hw_params_get_buffer_size(hw, ctypes.byref(size)), "buffer size")
            self.buffer_frames = size.value
            d = ctypes.c_int()
            self._check(lib.snd_pcm_hw_params_get_period_size(hw, ctypes.byref(size), ctypes.byref(d)), "period")
            self.period_frames = size.value
        finally:
            lib.snd_pcm_hw_params_free(hw)
        sw = ctypes.c_void_p()
        self._check(lib.snd_pcm_sw_params_malloc(ctypes.byref(sw)), "sw params")
        try:
            self._check(lib.snd_pcm_sw_params_current(pcm, sw), "sw params")
            boundary = ctypes.c_ulong()
            self._check(lib.snd_pcm_sw_params_get_boundary(sw, ctypes.byref(boundary)), "boundary")
            self._check(lib.snd_pcm_sw_params_set_start_threshold(pcm, sw, ctypes.c_ulong(1)), "start")
            self._check(lib.snd_pcm_sw_params_set_stop_threshold(pcm, sw, boundary), "stop threshold")
            self._check(lib.snd_pcm_sw_params_set_silence_threshold(pcm, sw, ctypes.c_ulong(0)), "silence")
            self._check(lib.snd_pcm_sw_params_set_silence_size(pcm, sw, boundary), "silence size")
            self._check(lib.snd_pcm_sw_params(pcm, sw), "sw params")
        finally:
            lib.snd_pcm_sw_params_free(sw)
        self._check(lib.snd_pcm_prepare(pcm), "prepare")
        self._write_all(bytes(self.period_frames * 4))          # starts the card: its one amplifier switch

    def _convert(self, data):
        if self.fmt == "S16_LE":
            return data
        out = bytearray(len(data) * 2)          # S32_LE: the 16-bit sample in the top two bytes
        out[2::4] = data[0::2]
        out[3::4] = data[1::2]
        return bytes(out)

    def _recover(self, err, what):
        log.warning("%s: %s - recovering", what, self.lib.snd_strerror(err).decode(errors="replace"))
        self._check(self.lib.snd_pcm_recover(self.pcm, err, 1), f"recover after {what}")

    def write(self, data):
        """Queue 16-bit stereo frames right after what is still playing (after silence: a period from now)."""
        avail = self.lib.snd_pcm_avail(self.pcm)
        if avail < 0:
            self._recover(avail, "avail")
        elif avail > self.buffer_frames:
            # free-running: the playing position ran past our last frame while ALSA played silence. Start the
            # phrase one period ahead of it (ALSA has filled that part with silence) instead of behind it
            n = self.lib.snd_pcm_forward(self.pcm, ctypes.c_ulong(avail - self.buffer_frames + self.period_frames))
            if n < 0:
                self._recover(n, "forward")
        return self._write_all(data)

    def _write_all(self, data):
        raw = self._convert(data)
        size = 4 * (2 if self.fmt == "S32_LE" else 1)
        frames, done = len(raw) // size, 0
        while done < frames:
            n = min(WRITE_FRAMES, frames - done)
            buf = (ctypes.c_char * (n * size)).from_buffer_copy(raw, done * size)
            got = self.lib.snd_pcm_writei(self.pcm, buf, ctypes.c_ulong(n))
            if got < 0:
                if got in (-EPIPE, -ESTRPIPE):
                    self._recover(got, "write")
                    continue
                self._check(got, "write")
            done += got
        return frames

    def delay(self):
        d = ctypes.c_long()
        err = self.lib.snd_pcm_delay(self.pcm, ctypes.byref(d))
        if err < 0:
            return 0
        return max(0, d.value)

    def close(self):
        if self.pcm:
            self.lib.snd_pcm_close(self.pcm)
            self.pcm = ctypes.c_void_p()


def open_pcm(device):
    """The PCM for this device name ('null' = no card). hw:N,M falls back to plughw:N,M."""
    if device == "null":
        return NullPcm()
    try:
        return AlsaPcm(device)
    except AlsaError as e:
        if device.startswith("hw:"):
            log.warning("%s (%s): trying plug%s", device, e, device)
            return AlsaPcm("plug" + device)
        raise


def read_exact(f, n):
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def say(out, line):
    out.write(line + "\n")
    out.flush()


def fade_out(data, frames=FADE_FRAMES):
    """The first `frames` of 16-bit stereo data, faded to silence (the end of a phrase cut short: no click)."""
    a = array("h")
    a.frombytes(data[:frames * 4 - (len(data[:frames * 4]) % 4)])
    n = len(a) // 2
    for i in range(n):
        g = 1.0 - (i + 1) / n
        a[2 * i] = int(a[2 * i] * g)
        a[2 * i + 1] = int(a[2 * i + 1] * g)
    return a.tobytes()


def serve(pcm, inp, out, clock=time.monotonic, sleep=time.sleep):
    """Play phrases from inp until EOF; report each one's end on out. Returns the exit code.
    STOP (someone talks over the robot): what is playing fades out within PIECE_FRAMES + the ALSA buffer
    (~0.2 s), everything sent before the STOP is dropped (each still reported "done"); the card keeps running."""
    say(out, f"ready {pcm.name} {pcm.fmt} buffer={pcm.buffer_frames} period={pcm.period_frames}")
    items = queue.Queue()
    gen = [0]                                   # STOPs so far: a phrase sent before the latest one is dropped

    def reader():
        while True:
            head = read_exact(inp, HEADER.size)
            if head is None:
                items.put(None)
                return
            magic, pid, nbytes = HEADER.unpack(head)
            if magic == STOP:
                gen[0] += 1
                continue
            if magic != MAGIC:
                items.put("bad")
                return
            data = read_exact(inp, nbytes)
            if data is None:
                items.put(None)
                return
            items.put((pid, data, gen[0]))
    threading.Thread(target=reader, name="speakerd-in", daemon=True).start()
    piece = PIECE_FRAMES * 4
    while True:
        item = items.get()
        if item is None:
            break
        if item == "bad":
            say(out, "error bad header")
            return 2
        pid, data, g = item
        data = data[:len(data) - len(data) % 4]
        try:
            for i in range(0, len(data), piece):
                if g != gen[0]:                 # stopped: a short fade instead of the rest
                    pcm.write(fade_out(data[i:]))
                    break
                pcm.write(data[i:i + piece])
        except AlsaError as e:
            say(out, f"error {e}")
            return 1
        say(out, f"done {pid} {clock() + pcm.delay() / RATE:.4f}")
    sleep(pcm.delay() / RATE)            # let the last phrase finish before the card closes
    return 0


def beep_frames(seconds=0.3, freq=660.0, peak=0.05):
    n = int(RATE * seconds)
    fade = int(0.01 * RATE)
    a = array("h")
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        v = int(peak * env * 32767 * math.sin(2 * math.pi * freq * i / RATE))
        a.extend((v, v))
    return a.tobytes()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="hw:0,0", help="ALSA device (default hw:0,0), or 'null' for no card")
    ap.add_argument("--card", type=int, help="same as --device hw:CARD,0")
    ap.add_argument("--beep", action="store_true", help="play one quiet beep (peak 0.05) and exit: a check")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s speakerd %(levelname)s %(message)s",
                        stream=sys.stderr)
    device = f"hw:{args.card},0" if args.card is not None else args.device
    try:
        pcm = open_pcm(device)
    except AlsaError as e:
        say(sys.stdout, f"error {e}")
        return 1
    log.info("open: %s %s, buffer %d frames, period %d", pcm.name, pcm.fmt, pcm.buffer_frames, pcm.period_frames)
    try:
        if args.beep:
            say(sys.stdout, f"ready {pcm.name} {pcm.fmt} buffer={pcm.buffer_frames} period={pcm.period_frames}")
            time.sleep(0.5)
            pcm.write(beep_frames())
            time.sleep(pcm.delay() / RATE + 0.5)
            return 0
        return serve(pcm, sys.stdin.buffer, sys.stdout)
    finally:
        pcm.close()
        log.info("closed")


if __name__ == "__main__":
    sys.exit(main())
