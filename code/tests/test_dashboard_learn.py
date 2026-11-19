"""pi/dashboard: the Learning card's API - /api/learn (learn.py's state.json) and /api/learn-decide (apply / reject a
proposal through code/learn.py)."""
import http.client
import importlib.util
import json
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

DASH = Path(__file__).resolve().parents[2] / "pi" / "dashboard" / "arcby_dashboard.py"
spec = importlib.util.spec_from_file_location("arcby_dashboard_learn", DASH)
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)


class LearnApiTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.old = dash.LEARN_STATE
        dash.LEARN_STATE = Path(self.dir.name) / "state.json"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        dash.LEARN_STATE = self.old
        self.dir.cleanup()

    def request(self, method, path, body=None, ctype="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        conn.request(method, path, body=body, headers={"Content-Type": ctype} if body is not None else {})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, json.loads(data) if data[:1] in (b"{", b"[") else data

    def test_no_state_yet(self):
        code, d = self.request("GET", "/api/learn")
        self.assertEqual(code, 200)
        self.assertIsNone(d["state"])
        self.assertIn("wall adapter", d["why"])

    def test_running_job_with_its_age(self):
        dash.LEARN_STATE.write_text(json.dumps(dict(running=True, job="faces", done=300, total=600, eta_s=240,
                                                    updated=time.time() - 1.5)))
        code, d = self.request("GET", "/api/learn")
        self.assertEqual(d["state"]["done"], 300)
        self.assertAlmostEqual(d["age_s"], 1.5, delta=1.0)

    def test_broken_state_file(self):
        dash.LEARN_STATE.write_text('{"running": tr')
        code, d = self.request("GET", "/api/learn")
        self.assertEqual(code, 200)
        self.assertIsNone(d["state"])
        self.assertIn("cannot read", d["why"])

    def test_decide_needs_json_and_a_proper_id(self):
        code, _ = self.request("POST", "/api/learn-decide", b'{"id": "p1", "accept": true}', ctype="text/plain")
        self.assertEqual(code, 415)
        for bad in (b'{"id": "p1; rm -rf ~", "accept": true}', b'{"id": "p1", "accept": "yes"}', b"[]"):
            code, d = self.request("POST", "/api/learn-decide", bad)
            self.assertEqual(code, 400, bad)

    def test_decide_runs_learn_py(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        res = dash.learn_decide(b'{"id": "p1793623000000", "accept": false}', run=run, cli="/x/learn.py")
        self.assertTrue(res["ok"])
        self.assertEqual(calls[0][1:], ["/x/learn.py", "reject", "p1793623000000"])

    def test_decide_reports_an_unknown_proposal(self):
        def run(cmd, **kw):
            return SimpleNamespace(returncode=1, stdout="", stderr="no proposal p5 (see: learn.py show)")
        res = dash.learn_decide(b'{"id": "p5", "accept": true}', run=run)
        self.assertFalse(res["ok"])
        self.assertIn("no proposal", res["message"])

    def test_decide_through_the_real_cli(self):
        """End to end: the dashboard -> learn.py apply on a real tuned.json (ARCBY_NO_PRIVATE keeps config clean)."""
        cli = Path(__file__).resolve().parents[1] / "learn.py"
        res = dash.learn_decide(b'{"id": "p1", "accept": true}', cli=cli,
                                run=lambda cmd, **kw: subprocess.run(cmd, **kw, env=dict(
                                    __import__("os").environ, HOME=self.dir.name)))
        self.assertFalse(res["ok"])                     # nothing proposed in an empty HOME
        self.assertIn("no proposal", res["message"])


if __name__ == "__main__":
    unittest.main()
