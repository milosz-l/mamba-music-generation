from typing import Iterable
from pathlib import Path

from miditok import MusicTokenizer
from symusic import Synthesizer ,dump_wav


def export_to_midi(tokenizer: MusicTokenizer, tokens: Iterable, filepath: Path | str):
    score = tokenizer.decode(tokens)
    score.dump_midi(filepath)


def export_to_wav(tokenizer: MusicTokenizer, tokens: Iterable, filepath: Path | str, sample_rate: int = 44100):
    score = tokenizer(tokens)
    synth = Synthesizer(
        sample_rate = sample_rate,
    )
    audio = synth.render(score, stereo=True)
    dump_wav(filepath, audio, sample_rate=sample_rate, use_int16=True)