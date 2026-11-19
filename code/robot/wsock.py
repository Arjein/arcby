"""
A minimal websocket client (RFC 6455), standard library only - enough for the Gemini Live API (robot/stt.py
LiveSTT): TLS, the HTTP upgrade, masked client frames, text / binary messages (fragmented too), ping -> pong, close.

    ws = WebSocket.connect("wss://host/path", headers={"x-goog-api-key": key}, timeout=10)
    ws.send_text(json.dumps(msg))
    text = ws.recv(timeout=5)       # str / bytes, None on timeout; raises Closed when the server closed
    ws.close()
"""
import base64
import os
import socket
import ssl
import struct
import threading
import urllib.parse


class Closed(Exception):
    """The connection is closed (by the server, or broken). .code / .reason from the close frame when there was one."""

    def __init__(self, code=None, reason=""):
        super().__init__(f"websocket closed ({code}: {reason})" if code else
                         f"websocket closed ({reason})" if reason else "websocket closed")
        self.code, self.reason = code, reason


OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


def frame(opcode, payload, mask_key=None):
    """One final client frame (masked, as RFC 6455 requires from clients)."""
    mask_key = os.urandom(4) if mask_key is None else mask_key
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 1 << 16:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    return head + mask_key + mask(payload, mask_key)


def mask(data, key):
    if not data:
        return b""
    k = (key * (len(data) // 4 + 1))[:len(data)]
    return (int.from_bytes(data, "big") ^ int.from_bytes(k, "big")).to_bytes(len(data), "big")


class WebSocket:
    def __init__(self, sock):
        self.sock = sock
        self._buf = b""
        self._send_lock = threading.Lock()
        self._parts, self._op = [], None      # a fragmented message being put together
        self.closed = False

    @classmethod
    def connect(cls, url, headers=None, timeout=10.0):
        u = urllib.parse.urlsplit(url)
        port = u.port or (443 if u.scheme == "wss" else 80)
        raw = socket.create_connection((u.hostname, port), timeout=timeout)
        sock = ssl.create_default_context().wrap_socket(raw, server_hostname=u.hostname) if u.scheme == "wss" else raw
        key = base64.b64encode(os.urandom(16)).decode()
        path = u.path + (f"?{u.query}" if u.query else "")
        lines = [f"GET {path} HTTP/1.1", f"Host: {u.hostname}", "Upgrade: websocket", "Connection: Upgrade",
                 f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
        lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        ws = cls(sock)
        head = ws._read_until(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 101 " not in status + " ":
            sock.close()
            raise Closed(None, f"upgrade refused: {status}")
        return ws

    def _read_until(self, end):
        while end not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise Closed(None, "connection ended")
            self._buf += chunk
        head, self._buf = self._buf.split(end, 1)
        return head

    def _read(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise Closed(None, "connection ended")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send(self, opcode, payload):
        if self.closed:
            raise Closed(None, "already closed")
        with self._send_lock:
            try:
                self.sock.sendall(frame(opcode, payload))
            except OSError as e:
                self.closed = True
                raise Closed(None, str(e)) from e

    def send_text(self, text):
        self._send(OP_TEXT, text.encode())

    def send_bytes(self, data):
        self._send(OP_BIN, bytes(data))

    def recv(self, timeout=None):
        """The next whole message (str for text, bytes for binary), None when nothing came within timeout."""
        if self.closed:
            raise Closed(None, "already closed")
        self.sock.settimeout(timeout)
        inside = False                      # a frame's header was read: a timeout now leaves half a frame behind
        try:
            while True:
                b0, b1 = self._read(2)
                inside = True
                self.sock.settimeout(30)      # the rest of a started frame comes right after
                op, fin, n = b0 & 0x0F, b0 & 0x80, b1 & 0x7F
                if n == 126:
                    n = struct.unpack(">H", self._read(2))[0]
                elif n == 127:
                    n = struct.unpack(">Q", self._read(8))[0]
                key = self._read(4) if b1 & 0x80 else None
                data = self._read(n)
                if key:
                    data = mask(data, key)
                if op == OP_PING:
                    self._send(OP_PONG, data)
                elif op == OP_CLOSE:
                    self.closed = True
                    code = struct.unpack(">H", data[:2])[0] if len(data) >= 2 else None
                    with self._send_lock:
                        try:
                            self.sock.sendall(frame(OP_CLOSE, data[:2]))
                        except OSError:
                            pass
                    self.sock.close()
                    raise Closed(code, data[2:].decode(errors="replace"))
                elif op in (OP_TEXT, OP_BIN, OP_CONT):
                    if op != OP_CONT:
                        self._op, self._parts = op, []
                    self._parts.append(data)
                    if fin:
                        msg = b"".join(self._parts)
                        self._parts = []
                        return msg.decode() if self._op == OP_TEXT else msg
                inside = False
                self.sock.settimeout(timeout)
        except (socket.timeout, TimeoutError):
            if inside:                      # the next recv would start in the middle of a frame: unusable
                self.closed = True
                try:
                    self.sock.close()
                except OSError:
                    pass
                raise Closed(None, "timed out inside a frame")
            return None
        except OSError as e:
            self.closed = True
            raise Closed(None, str(e)) from e

    def close(self):
        if self.closed:
            return
        try:
            self._send(OP_CLOSE, struct.pack(">H", 1000))
        except Closed:
            pass
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass
