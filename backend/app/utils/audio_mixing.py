import asyncio
from pathlib import Path


NARRATION_MUSIC_INTRO_SECONDS = 2.5
NARRATION_MUSIC_OUTRO_SECONDS = 5.0
NARRATION_MUSIC_LEVEL = 0.25


def narration_music_duration(voice_duration: float) -> float:
    """Return the music duration needed for the narration intro and outro."""
    return (
        voice_duration
        + NARRATION_MUSIC_INTRO_SECONDS
        + NARRATION_MUSIC_OUTRO_SECONDS
    )


async def mix_narration_and_music(
    voice_path: Path,
    music_path: Path,
    output_path: Path,
    intro_seconds: float = NARRATION_MUSIC_INTRO_SECONDS,
    outro_seconds: float = NARRATION_MUSIC_OUTRO_SECONDS,
    music_level: float = NARRATION_MUSIC_LEVEL,
) -> Path:
    """Layer narration over a longer music bed with a faded intro and outro."""
    import numpy as np
    import soundfile as sf

    def _mix():
        voice, voice_sr = sf.read(str(voice_path), dtype="float32")
        music, music_sr = sf.read(str(music_path), dtype="float32")

        # Resample music to match voice sample rate if needed
        if music_sr != voice_sr:
            indices = np.round(np.arange(0, len(music), music_sr / voice_sr)).astype(int)
            indices = indices[indices < len(music)]
            music = music[indices]

        # Ensure both are 1D (mono) for simple mixing
        if voice.ndim > 1:
            voice = voice.mean(axis=1)
        if music.ndim > 1:
            music = music.mean(axis=1)

        if len(music) == 0:
            raise ValueError("Background music file contains no audio samples")

        intro_len = max(0, round(intro_seconds * voice_sr))
        outro_len = max(0, round(outro_seconds * voice_sr))
        target_len = intro_len + len(voice) + outro_len

        # Delay the narration so the music can establish the broadcast first,
        # then leave the music alone for a short outro after the voice ends.
        voice = np.pad(voice, (intro_len, outro_len))

        # Match the generated music to the complete broadcast timeline.
        if len(music) < target_len:
            repeats = (target_len // len(music)) + 1
            music = np.tile(music, repeats)[:target_len]
        else:
            music = music[:target_len]

        # Give the music a full-length fade during the intro and outro. Clamp
        # the fades to half the file so unusually short inputs remain valid.
        max_fade_len = target_len // 2
        fade_in_len = min(intro_len, max_fade_len)
        fade_out_len = min(outro_len, max_fade_len)
        if fade_in_len:
            fade_in = np.sin(np.linspace(0, np.pi / 2, fade_in_len)) ** 2
            music[:fade_in_len] *= fade_in
        if fade_out_len:
            fade_out = np.cos(np.linspace(0, np.pi / 2, fade_out_len)) ** 2
            music[-fade_out_len:] *= fade_out

        # Keep narration at full level and place the music underneath it.
        mixed = voice + music * music_level
        # Normalize to prevent clipping
        peak = np.abs(mixed).max()
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), mixed, samplerate=voice_sr)

    await asyncio.to_thread(_mix)
    return output_path
