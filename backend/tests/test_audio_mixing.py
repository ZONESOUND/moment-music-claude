import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.utils.audio_mixing import (
    mix_narration_and_music,
    narration_music_duration,
)


class AudioMixingTests(unittest.IsolatedAsyncioTestCase):
    async def test_adds_faded_music_intro_and_outro(self):
        sample_rate = 1_000
        voice_seconds = 1.0
        intro_seconds = 0.2
        outro_seconds = 0.3
        voice = np.full(round(voice_seconds * sample_rate), 0.5, dtype=np.float32)
        music = np.full(
            round((voice_seconds + intro_seconds + outro_seconds) * sample_rate),
            0.4,
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            voice_path = root / "voice.wav"
            music_path = root / "music.wav"
            output_path = root / "mixed.wav"
            sf.write(voice_path, voice, sample_rate, subtype="FLOAT")
            sf.write(music_path, music, sample_rate, subtype="FLOAT")

            await mix_narration_and_music(
                voice_path,
                music_path,
                output_path,
                intro_seconds=intro_seconds,
                outro_seconds=outro_seconds,
            )

            mixed, mixed_sr = sf.read(output_path, dtype="float32")

        self.assertEqual(mixed_sr, sample_rate)
        self.assertEqual(len(mixed), len(music))
        self.assertAlmostEqual(float(mixed[0]), 0.0, places=5)
        self.assertGreater(float(mixed[round(intro_seconds * sample_rate)]), 0.5)
        self.assertAlmostEqual(float(mixed[-1]), 0.0, places=5)

    def test_music_duration_includes_default_intro_and_outro(self):
        self.assertEqual(narration_music_duration(12.5), 20.0)


if __name__ == "__main__":
    unittest.main()
