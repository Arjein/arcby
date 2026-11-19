"""pi/dashboard camera live view: JPEG splitting, start/stop of the camera program, the MJPEG stream.
A small Python script stands in for rpicam-vid."""
import http.client
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

DASH = Path(__file__).resolve().parents[2] / "pi" / "dashboard" / "arcby_dashboard.py"
spec = importlib.util.spec_from_file_location("arcby_dashboard", DASH)
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)

JPEG = b"\xff\xd8" + b"\x12\xff\x00\x34" * 50 + b"\xff\xd9"

# Fake rpicam-vid: JPEG frames on stdout at ~50 fps, split into odd-sized writes.
FAKE_CAMERA = r"""
import os, sys, time
frame = b"\xff\xd8" + b"\x12\xff\x00\x34" * 50 + b"\xff\xd9"
out = sys.stdout.buffer
while True:
    out.write(frame[:7]); out.flush()
    out.write(frame[7:]); out.flush()
    time.sleep(0.02)
"""
BUSY_CAMERA = r"""
import sys
print("[0:01:02] INFO Camera camera_manager.cpp:330 libcamera v0.5", file=sys.stderr)
print("ERROR: *** failed to acquire camera /base/soc/i2c0mux/i2c@1/imx708@1a ***", file=sys.stderr)
sys.exit(255)
"""


def streamer(script, idle_s=0.1):
    return dash.CameraStreamer(cmd=[sys.executable, "-c", script], idle_s=idle_s)


def wait_until(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


class SplitTest(unittest.TestCase):
    def test_two_frames_and_a_partial(self):
        buf = bytearray(JPEG + JPEG + JPEG[:10])
        self.assertEqual(dash.split_jpegs(buf), [JPEG, JPEG])
        self.assertEqual(bytes(buf), JPEG[:10])
        buf += JPEG[10:]
        self.assertEqual(dash.split_jpegs(buf), [JPEG])
        self.assertEqual(buf, bytearray())

    def test_junk_before_a_frame_is_dropped(self):
        buf = bytearray(b"junk" + JPEG)
        self.assertEqual(dash.split_jpegs(buf), [JPEG])

    def test_half_start_marker_kept(self):
        buf = bytearray(b"junk\xff")
        self.assertEqual(dash.split_jpegs(buf), [])
        self.assertEqual(bytes(buf), b"\xff")
        buf += JPEG[1:]
        self.assertEqual(dash.split_jpegs(buf), [JPEG])


class StreamerTest(unittest.TestCase):
    def test_runs_only_while_watched(self):
        cam = streamer(FAKE_CAMERA)
        self.assertIsNone(cam.proc)
        cam.open_viewer()
        got = cam.wait_frame(0, 5.0)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], JPEG)
        self.assertEqual(cam.snapshot()["status"], "live")
        proc = cam.proc
        cam.close_viewer()
        self.assertTrue(wait_until(lambda: proc.poll() is not None))
        self.assertEqual(cam.snapshot()["status"], "off")
        self.assertEqual(cam.snapshot()["error"], "")

    def test_second_viewer_shares_the_camera(self):
        cam = streamer(FAKE_CAMERA)
        cam.open_viewer()
        proc = cam.proc
        cam.open_viewer()
        self.assertIs(cam.proc, proc)
        cam.close_viewer()
        time.sleep(0.3)                          # longer than idle_s: one viewer is still there
        self.assertIsNone(proc.poll())
        cam.close_viewer()
        self.assertTrue(wait_until(lambda: proc.poll() is not None))

    def test_busy_camera_reports_the_error(self):
        cam = streamer(BUSY_CAMERA)
        cam.open_viewer()
        self.assertIsNone(cam.wait_frame(0, 5.0))
        snap = cam.snapshot()
        self.assertEqual(snap["status"], "error")
        self.assertIn("failed to acquire camera", snap["error"])
        cam.close_viewer()

    def test_missing_program(self):
        cam = dash.CameraStreamer(cmd=["/nonexistent/rpicam-vid"])
        cam.open_viewer()
        self.assertIsNone(cam.wait_frame(0, 1.0))
        self.assertEqual(cam.snapshot()["status"], "error")
        cam.close_viewer()


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.cam = streamer(FAKE_CAMERA)
        self.old, dash._camera = dash._camera, self.cam
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        dash._camera = self.old

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        conn.request("GET", path)
        return conn, conn.getresponse()

    def test_mjpeg_stream(self):
        conn, r = self.get("/camera.mjpg")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Content-Type"), "multipart/x-mixed-replace; boundary=frame")
        data = b""
        while data.count(b"--frame") < 3:
            data += r.read1(65536)
        part = data.split(b"--frame\r\n")[1]
        head, _, body = part.partition(b"\r\n\r\n")
        self.assertIn(f"Content-Length: {len(JPEG)}".encode(), head)
        self.assertTrue(body.startswith(JPEG))
        self.assertEqual(self.cam.viewers, 1)
        r.close()
        conn.close()
        self.assertTrue(wait_until(lambda: self.cam.viewers == 0))
        self.assertTrue(wait_until(lambda: self.cam.snapshot()["status"] == "off"))

    def test_sensors_json_has_camera_status(self):
        conn, r = self.get("/api/sensors")
        body = json.loads(r.read())
        conn.close()
        self.assertEqual(body["camera"]["status"], "off")

    def test_busy_camera_gives_503_with_reason(self):
        dash._camera = streamer(BUSY_CAMERA)
        conn, r = self.get("/camera.mjpg")
        self.assertEqual(r.status, 503)
        self.assertIn(b"failed to acquire camera", r.read())
        conn.close()

    def test_page_has_camera_card(self):
        conn, r = self.get("/")
        page = r.read().decode()
        conn.close()
        for needle in ('id="cambtn"', 'id="camimg"', "camera.mjpg"):
            self.assertIn(needle, page)


@unittest.skipUnless(shutil.which("node"), "needs node")
class PageScriptTest(unittest.TestCase):
    def test_page_script_parses(self):
        script = re.search(r"<script>(.*)</script>", dash.PAGE, re.S).group(1)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
        try:
            r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
        finally:
            Path(f.name).unlink()
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()


class SpeakerLevelTest(unittest.TestCase):
    """The dashboard's one write: the speaker level file (0-100 % in, 0-1 stored), nothing else accepted."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.old = dash.LEVEL_FILE
        dash.LEVEL_FILE = Path(self.dir.name) / "speaker_level.json"

    def tearDown(self):
        dash.LEVEL_FILE = self.old
        self.dir.cleanup()

    def test_set_and_read_back(self):
        self.assertEqual(dash.speaker_level(), {"level": None})
        self.assertEqual(dash.set_speaker_level(b'{"level": 35}'), {"level": 0.35})
        self.assertEqual(json.loads(dash.LEVEL_FILE.read_text()), {"level": 0.35})

    def test_out_of_range_or_junk_is_refused(self):
        for body in (b'{"level": 150}', b'{"level": -5}', b'{"x": 1}', b"not json"):
            with self.assertRaises((ValueError, KeyError)):
                dash.set_speaker_level(body)
        self.assertFalse(dash.LEVEL_FILE.exists())

    def test_only_json_posts_change_the_level(self):
        """Another web page can send text/plain without asking (no CORS preflight) - refused."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for ctype, expect in (("text/plain", 415), ("application/json", 200)):
                c = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                c.request("POST", "/api/speaker-level", body=b'{"level": 20}', headers={"Content-Type": ctype})
                self.assertEqual(c.getresponse().status, expect)
                c.close()
            self.assertEqual(json.loads(dash.LEVEL_FILE.read_text()), {"level": 0.2})
        finally:
            server.shutdown()
            server.server_close()


class RobotRunningTest(unittest.TestCase):
    def test_finds_alive_py_in_proc(self):
        with tempfile.TemporaryDirectory() as d:
            for pid, cmd in (("12", b"bash\0"), ("34", b"python3\0code/alive.py\0--quiet\0")):
                Path(d, pid).mkdir()
                Path(d, pid, "cmdline").write_bytes(cmd)
            self.assertTrue(dash.robot_running(d))
            Path(d, "34", "cmdline").write_bytes(b"python3\0other.py\0")
            self.assertFalse(dash.robot_running(d))
