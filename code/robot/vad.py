"""
Voice activity detection: is someone speaking? Silero VAD v5 (MIT, robot/faces.py MODELS["silero_vad"]) on
onnxruntime, ~0.5 ms per 32 ms chunk on the Pi - so the robot only ever looks at sound that contains speech.

    vad = Vad()                      # models from config.MODELS_DIR
    p = vad.prob(chunk)              # chunk = 512 float32 samples at 16 kHz (-1..1) -> 0..1
    vad.reset()                      # between unrelated recordings

The v5 model keeps a state between chunks and wants the last 64 samples of the previous chunk in front of each new
one (as in silero-vad's own OnnxWrapper).
"""
import numpy as np

from . import faces

RATE, CHUNK, CONTEXT = 16000, 512, 64


class Vad:
    def __init__(self, models_dir=None, threads=1):
        self.session = faces._session(faces.model_path("silero_vad", models_dir), threads)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, CONTEXT), np.float32)

    def prob(self, chunk):
        x = np.asarray(chunk, np.float32).reshape(1, CHUNK)
        inp = np.concatenate([self.context, x], axis=1)
        out, self.state = self.session.run(None, {"input": inp, "state": self.state, "sr": np.array(RATE, np.int64)})
        self.context = x[:, -CONTEXT:]
        return float(out[0, 0])

    def probs(self, pcm):
        """Speech probability per chunk of a whole 16 kHz recording (resets first)."""
        self.reset()
        n = len(pcm) // CHUNK
        return [self.prob(pcm[i * CHUNK:(i + 1) * CHUNK]) for i in range(n)]
