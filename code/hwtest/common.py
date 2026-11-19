"""
Shared helpers for the Arcby hardware tests: operator prompts, a result
report that is printed and saved, power check and safe pin release.
"""
import os
import select
import subprocess
import sys
import termios
import tty
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOG_DIR = Path.home() / "scratch" / "hwtest"

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"


# ---------- keyboard ----------

@contextmanager
def cbreak():
    """Single key presses without Enter; Ctrl+C still works."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def flush_input():
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def read_key():
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()


def getkey(prompt, choices="yn"):
    """Wait for one key from `choices` and return it."""
    flush_input()
    print(f"{prompt} [{'/'.join(choices)}] ", end="", flush=True)
    with cbreak():
        while True:
            k = read_key()
            if k in choices:
                print(k)
                return k


def key_pressed(timeout):
    """True if a key was pressed within `timeout` seconds. Use inside cbreak()."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        read_key()
        return True
    return False


def ask_text(prompt):
    flush_input()
    return input(f"{prompt} ").strip()


def heading(text):
    print(f"\n=== {text} ===")


# ---------- system ----------

def require_battery(what="the motors"):
    """Stop the test (SystemExit with the reason) when the battery is off - robot/power.py's read-only check."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from robot.power import BatteryOff, require_battery as req
    try:
        req(what)
    except BatteryOff as e:
        raise SystemExit(str(e)) from None


def check_power():
    """(ok, text) from `vcgencmd get_throttled`. Bit 0 = under-voltage now,
    bit 16 = under-voltage happened since boot."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        value = int(out.split("=")[1], 16)
    except Exception as e:
        return None, f"could not read ({e})"
    if value & 0x1:
        return False, f"{out}: Pi is UNDER-VOLTAGE right now (check the Pi buck, 5.1 V)"
    if value & 0x10000:
        return False, f"{out}: under-voltage happened since boot (Pi buck sagged)"
    return True, out


def release_pins(pins):
    """Leave pins as inputs with pull-down, so nothing downstream sees a
    floating input after the test (gpiozero's close() leaves them unpulled)."""
    if os.environ.get("GPIOZERO_PIN_FACTORY") == "mock":
        return
    subprocess.run(["pinctrl", "set", ",".join(map(str, pins)), "ip", "pd"],
                   capture_output=True)


# ---------- speech (quiet) ----------

def say(text, level=0.08, voice="en-gb", speed=150, card=0):
    """Speak through the I2S speaker with espeak-ng. The card has no volume control, so loudness = the
    sample peak (level, full scale = 1; keep <= 0.08 at night - the household sleeps)."""
    import io
    import wave

    import numpy as np
    wav = subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "--stdout", text],
                         capture_output=True, check=True).stdout
    w = wave.open(io.BytesIO(wav))
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    x = x / (np.abs(x).max() or 1) * level
    n = int(len(x) * 48000 / w.getframerate())             # resample to the I2S rate, stereo
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    buf = io.BytesIO()
    o = wave.open(buf, "wb")
    o.setnchannels(2)
    o.setsampwidth(2)
    o.setframerate(48000)
    o.writeframes((np.repeat(y, 2) * 32767).astype(np.int16).tobytes())
    o.close()
    subprocess.run(["aplay", "-q", "-D", f"plughw:{card},0", "-"], input=buf.getvalue(), check=True)


# ---------- listening (faster-whisper in its own venv) ----------

STT_PY = Path.home() / ".venvs" / "arcby-stt" / "bin" / "python"
STT_SERVER = r"""
import sys
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[1], device="cpu", compute_type="int8")
print("ready", flush=True)
for line in sys.stdin:
    path, lang, prompt = line.rstrip("\n").split("\t")
    segs, _ = m.transcribe(path, language=lang or None, beam_size=1, vad_filter=True,
                           initial_prompt=prompt or None)
    print(" ".join(s.text.strip() for s in segs).replace("\n", " "), flush=True)
"""


class Listener:
    """Record a few seconds from the I2S mic and turn them into text. The model stays loaded in a child
    process (the venv has faster-whisper; loading it takes seconds)."""

    def __init__(self, model="base", card=0, workdir=LOG_DIR / "listen"):
        self.card, self.dir = card, Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen([str(STT_PY), "-u", "-c", STT_SERVER, model], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        if self.proc.stdout.readline().strip() != "ready":
            raise RuntimeError("speech recognition did not start")

    def hear(self, seconds, lang="tr", prompt=""):
        """(text, rms) of the next `seconds` of sound."""
        import wave

        import numpy as np
        raw, mono = self.dir / "heard.wav", self.dir / "heard16k.wav"
        subprocess.run(["arecord", "-q", "-D", f"plughw:{self.card},0", "-c", "2", "-r", "48000", "-f", "S32_LE",
                        "-d", str(int(seconds)), str(raw)], check=True)
        w = wave.open(str(raw))
        d = np.frombuffer(w.readframes(w.getnframes()), np.int32).reshape(-1, 2)[:, 0].astype(float) / 2**31
        d -= d.mean()
        rms = float(np.sqrt((d ** 2).mean()))
        m = d[::3]                                         # 48 kHz -> 16 kHz (speech band only)
        m = m / (np.abs(m).max() or 1) * 0.9
        o = wave.open(str(mono), "wb")
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(16000)
        o.writeframes((m * 32767).astype(np.int16).tobytes())
        o.close()
        self.proc.stdin.write(f"{mono}\t{lang}\t{prompt}\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip(), rms

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


TR_PLAIN = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def plain_tr(text):
    """Turkish text -> lower-case ASCII words ("Hazırım!" -> "hazirim")."""
    t = text.replace("I", "ı").replace("İ", "i").lower().translate(TR_PLAIN)
    return "".join(c if "a" <= c <= "z" else " " for c in t).split()


def heard_any(text, words, cutoff=0.75):
    """Does `text` contain one of `words` (plain ASCII), allowing for a slightly wrong transcription?"""
    import difflib
    for w in plain_tr(text):
        for k in words:
            if w.startswith(k) or difflib.SequenceMatcher(None, w, k).ratio() >= cutoff:
                return k
    return None


# ---------- report ----------

class Report:
    def __init__(self, unit):
        self.unit = unit
        self.started = datetime.now()
        self.rows = []    # (test, result, detail)
        self.hints = []   # what to check, derived from failures
        self.facts = []   # measured values / operator inputs

    def add(self, test, result, detail=""):
        self.rows.append((test, result, detail))
        print(f"   [{result}] {test}" + (f" - {detail}" if detail else ""))

    def hint(self, text):
        if text not in self.hints:
            self.hints.append(text)

    def fact(self, text):
        self.facts.append(text)

    @property
    def failed(self):
        return any(r == FAIL for _, r, _ in self.rows)

    def text(self):
        lines = [f"Arcby hardware test: {self.unit}",
                 f"Started: {self.started:%Y-%m-%d %H:%M:%S}", ""]
        width = max((len(t) for t, _, _ in self.rows), default=10)
        for test, result, detail in self.rows:
            lines.append(f"  {result:4}  {test:{width}}  {detail}")
        if self.facts:
            lines += ["", "Measured / noted:"] + [f"  - {f}" for f in self.facts]
        if self.hints:
            lines += ["", "What to check:"] + [f"  - {h}" for h in self.hints]
        verdict = "FAILED" if self.failed else "ALL PASSED"
        lines += ["", f"Result: {verdict}"]
        return "\n".join(lines)

    def save(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{self.unit}-{self.started:%Y%m%d-%H%M%S}.txt"
        path.write_text(self.text() + "\n")
        return path
