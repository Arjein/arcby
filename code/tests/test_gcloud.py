"""robot/gcloud.py: service-account login (JWT signed by openssl, token cached)."""
import base64
import json
import tempfile
import unittest
from pathlib import Path

from robot.gcloud import AccessToken, make_jwt


class GcloudTest(unittest.TestCase):
    def test_jwt_claims(self):
        info = {"client_email": "a@b", "token_uri": "https://t", "private_key": "KEY"}
        jwt = make_jwt(info, 1000, sign=lambda m, k: b"sig")
        head, claims, sig = jwt.split(".")
        c = json.loads(base64.urlsafe_b64decode(claims + "=="))
        self.assertEqual((c["iss"], c["aud"], c["iat"], c["exp"]), ("a@b", "https://t", 1000, 4600))
        self.assertEqual(base64.urlsafe_b64decode(sig + "=="), b"sig")

    def test_token_cached_and_no_key_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            key = Path(d) / "k.json"
            self.assertIsNone(AccessToken(key, post=lambda u, b: {}).get())
            key.write_text(json.dumps({"client_email": "a@b", "token_uri": "https://t", "private_key": "K"}))
            calls = []
            t = AccessToken(key, clock=lambda: 1000.0, sign=lambda m, k: b"s",
                            post=lambda u, b: calls.append(u) or {"access_token": "ya29", "expires_in": 3600})
            self.assertEqual(t.get(), "ya29")
            self.assertEqual(t.get(), "ya29")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
