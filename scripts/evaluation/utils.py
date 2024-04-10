from pathlib import Path

import numpy as np
from music21 import midi


def open_midi(midi_path: Path, remove_drums: bool=False):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]          
            
    return midi.translate.midiFileToStream(mf)


def kl_divergence(p: np.ndarray, q: np.ndarray):
    div = (p * np.log(p / q)).sum()
    return div