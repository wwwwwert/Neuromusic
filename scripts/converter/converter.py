import os
import random
import subprocess
from pathlib import Path
from typing import Optional

from miditok.midi_tokenizer import MIDITokenizer
from symusic.core import ScoreTick
from torch import Tensor
from torchaudio import load
from torchaudio.functional import resample

from scripts.utils.download import download


class Converter:
    def __init__(
        self, 
        soundfont: Optional[str]=None, 
        sample_rate: int=16000, 
        tokenizer: Optional[MIDITokenizer]=None
    ) -> None:
        if soundfont is None:
            soundfont = 'scripts/converter/Touhou.sf2'
            if not Path(soundfont).exists():
                download(
                    'https://musical-artifacts.com/artifacts/433/Touhou.sf2', 
                    'scripts/converter/Touhou.sf2'
                )

        self.fs = FluidSynth(sound_font=soundfont, sample_rate=sample_rate)
        self.tokenizer = tokenizer

    def midi_to_audio(
        self,
        midi_path: str,
        file_path: str,
    ):
        """Converts midi to mp3.

        Args:
            midi_path: path to midi
            file_name: file to save output
        """
        save_dir = os.path.dirname(file_path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        self.fs.midi_to_audio(midi_path, file_path)

    def midi_to_tensor(
        self,
        midi_path: str,
        mono: bool=True
    ):
        os.makedirs('scripts/converter/tmp_audios', exist_ok=True)
        hash = random.getrandbits(128)
        name = "%032x" % hash
        audio_path = f'scripts/converter/tmp_audios/{name}.wav'
        self.midi_to_audio(midi_path, audio_path)
        audio_tensor, sr = load(audio_path)
        os.remove(audio_path)
        if mono:
            audio_tensor = audio_tensor.sum(dim=0)
        return audio_tensor


    def score_to_audio(
        self,
        score: ScoreTick,
        file_path: str,
    ):
        os.makedirs('scripts/converter/tmp_midis', exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(file_path))
        midi_path = f'scripts/converter/tmp_midis/{name}.midi'
        score.dump_midi(midi_path)
        self.midi_to_audio(midi_path, file_path)
        os.remove(midi_path)

    def score_to_tensor(
        self,
        score: ScoreTick,
        mono: bool=True
    ):
        os.makedirs('scripts/converter/tmp_audios', exist_ok=True)
        hash = random.getrandbits(128)
        name = "%032x" % hash
        audio_path = f'scripts/converter/tmp_audios/{name}.wav'
        self.score_to_audio(score, audio_path)
        audio_tensor, sr = load(audio_path)
        os.remove(audio_path)
        if mono:
            audio_tensor = audio_tensor.sum(dim=0)
        return audio_tensor


class FluidSynth():
    def __init__(self, sound_font, sample_rate):
        self.sample_rate = sample_rate
        self.sound_font = os.path.expanduser(sound_font)

    def midi_to_audio(self, midi_file, audio_file):
        subprocess.call(
            ['fluidsynth', '-ni', self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)], 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
