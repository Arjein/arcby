"""
Google Cloud login with a service account (config.GCLOUD_KEY_FILE, a JSON key - a secret: chmod 600, never in git,
never printed) -> an OAuth access token, standard library + the `openssl` command (signs the JWT, RS256). Cloud
Text-to-Speech does not take API keys (HTTP 401, 2026-11-01).

    token = AccessToken()            # .get() -> "ya29...." (cached ~55 min), None when there is no key file
"""
import base64
import json
import logging
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def sign_rs256(message, private_key_pem, run=subprocess.run):
    """RS256 signature of message (bytes) with a PEM private key, by `openssl dgst -sha256 -sign` (key file in a
    private temp dir, removed at once)."""
    with tempfile.TemporaryDirectory() as d:
        key = Path(d) / "k.pem"
        key.touch(mode=0o600)
        key.write_text(private_key_pem)
        out = run(["openssl", "dgst", "-sha256", "-sign", str(key)], input=message, capture_output=True,
                  check=True, timeout=10)
    return out.stdout


def make_jwt(info, now, scope=SCOPE, sign=sign_rs256):
    head = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64(json.dumps({"iss": info["client_email"], "scope": scope, "aud": info["token_uri"],
                              "iat": int(now), "exp": int(now) + 3600}).encode())
    signing = f"{head}.{claims}".encode()
    return f"{head}.{claims}.{_b64(sign(signing, info['private_key']))}"


class AccessToken:
    def __init__(self, key_file=None, clock=time.time, post=None, sign=sign_rs256):
        self.key_file = Path(key_file or config.GCLOUD_KEY_FILE).expanduser()
        self.clock, self.sign = clock, sign
        self.post = post or self._post
        self._token, self._until = None, 0.0
        self._lock = threading.Lock()
        self.error = None

    @property
    def available(self):
        return self.key_file.exists()

    def get(self):
        """A valid access token, or None (no key file / login failed - .error says why)."""
        with self._lock:
            if self._token and self.clock() < self._until - 300:
                return self._token
            if not self.available:
                self.error = "no key file"
                return None
            try:
                info = json.loads(self.key_file.read_text())
                jwt = make_jwt(info, self.clock(), sign=self.sign)
                body = urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                               "assertion": jwt}).encode()
                got = self.post(info["token_uri"], body)
                self._token = got["access_token"]
                self._until = self.clock() + float(got.get("expires_in", 3600))
                self.error = None
                return self._token
            except Exception as e:  # noqa: BLE001 - no login: the backup voice is simply not there
                self.error = f"{type(e).__name__}: {str(e)[:120]}"
                log.warning("gcloud: login failed (%s)", self.error)
                return None

    @staticmethod
    def _post(url, body):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
