import os
from typing import List

import gensim
import numpy as np
from music21 import analysis, chord, roman, stream
from music21.stream.base import Score as m21_score

from scripts.base.base_feature import FeatureBase

from .utils import kl_divergence


class DurationFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Duration Feature')

    def __call__(self, score: m21_score):
        return score.duration.quarterLength
    
    def distance(self, value_1, value_2):
        return value_1 - value_2


class KeyFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Key Feature')

    def __call__(self, score: m21_score):
        return score.analyze('key')
    
    def distance(self, value_1, value_2):
        return 0 if value_1 == value_2 else 1
    

class PitchClassDistributionFeature(FeatureBase):
    pitches_range = [
        'C',
        'C#',
        'D',
        'D#',
        'E',
        'F',
        'F#',
        'G',
        'G#',
        'A', 
        'A#', 
        'B'
    ]

    def __init__(self) -> None:
        super().__init__('Pitch Class Distribution Feature')
    
    def __call__(self, score: m21_score):
        pitch_count = analysis.pitchAnalysis.pitchAttributeCount(score, 'name')
        res = []
        for pitch in self.pitches_range:
            if pitch in pitch_count:
                res.append(pitch_count[pitch])
            else:
                res.append(0)
        res = np.array(res)
        res = res / res.sum()
        return list(res)
    
    def distance(self, value_1, value_2):
        value_1 = np.array(value_1) + 1e-6
        value_2 = np.array(value_2) + 1e-6
        return kl_divergence(value_1, value_2)
    

class RhythmFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Rhythm Feature')

    def __call__(self, score: m21_score):
        time_signatures = score.getTimeSignatures()
        rhythms = []
        for ts in time_signatures:
            r = "{0}/{1}".format(ts.beatCount, ts.denominator)
            if not rhythms or rhythms[-1] != r:
                rhythms.append(r)
        return rhythms
    
    def distance(self, value_1, value_2):
        return 0 if value_1[-1] == value_2[-1] else 1


class HarmonicReductionFeature(FeatureBase):
    def __init__(self, max_notes_per_chord: int=4, model_path: str=None):
        self.max_notes_per_chord = max_notes_per_chord
        self.model = None
        if model_path is not None:
            self.model = gensim.models.word2vec.Word2Vec.load(model_path)
        super().__init__('Harmonic Reduction Feature')

    def __call__(self, score: m21_score):
        ret = []
        temp_midi = stream.Score()
        temp_midi_chords = score.chordify()
        temp_midi.insert(0, temp_midi_chords)    
        music_key = temp_midi.analyze('key')
        max_notes_per_chord = self.max_notes_per_chord
        for m in temp_midi_chords.measures(0, None): # None = get all measures.
            if (type(m) != stream.Measure):
                continue
            
            # Here we count all notes length in each measure,
            # get the most frequent ones and try to create a chord with them.
            count_dict = dict()
            bass_note = self.note_count(m, count_dict)
            if (len(count_dict) < 1):
                ret.append("-") # Empty measure
                continue
            
            sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
            sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
            measure_chord = chord.Chord(sorted_notes)
            
            # Convert the chord to the functional roman representation
            # to make its information independent of the music key.
            roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
            ret.append(self.simplify_roman_name(roman_numeral))
            
        return ret

    @staticmethod
    def note_count(measure, count_dict):
        bass_note = None
        for chord in measure.recurse().getElementsByClass('Chord'):
            # All notes have the same length of its chord parent.
            note_length = chord.quarterLength
            for note in chord.pitches:          
                # If note is "C5", note.name is "C". We use "C5"
                # style to be able to detect more precise inversions.
                note_name = str(note) 
                if (bass_note is None or bass_note.ps > note.ps):
                    bass_note = note
                    
                if note_name in count_dict:
                    count_dict[note_name] += note_length
                else:
                    count_dict[note_name] = note_length

        return bass_note
    
    @staticmethod
    def simplify_roman_name(roman_numeral):
        # Chords can get nasty names as "bII#86#6#5",
        # in this method we try to simplify names, even if it ends in
        # a different chord to reduce the chord vocabulary and display
        # chord function clearer.
        ret = roman_numeral.romanNumeral
        inversion_name = None
        inversion = roman_numeral.inversion()
        
        # Checking valid inversions.
        if ((roman_numeral.isTriad() and inversion < 3) or
                (inversion < 4 and
                    (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
            inversion_name = roman_numeral.inversionName()
            
        if (inversion_name is not None):
            ret = ret + str(inversion_name)
        elif (roman_numeral.isDominantSeventh()): 
            ret = ret + "M7"
        elif (roman_numeral.isDiminishedSeventh()): 
            ret = ret + "o7"
        return ret

    def distance(self, value_1, value_2):
        if self.model is None:
            raise AttributeError('You need to train or add pretrained Word2Vec model.')
        vec_1 = self.vectorize_harmony(value_1)
        vec_2 = self.vectorize_harmony(value_2)

        return self.cosine_similarity(vec_1, vec_2)

    @staticmethod
    def cosine_similarity(vecA, vecB):
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def vectorize_harmony(self, harmonic_reduction: List[str]):
        word_vecs = []
        for word in harmonic_reduction:
            # if word in self.model.wv.key_to_index:
            vec = self.model.wv[word]
            word_vecs.append(vec)
        return np.mean(word_vecs, axis=0)
    

class NotesDurationDistributionFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Notes Duration Distribution Feature')

    def __call__(self, score: m21_score):
        notes_lengths_distribution = {}
        for note in score.flat.notes:
            note_length = note.quarterLength
            if note_length not in notes_lengths_distribution:
                notes_lengths_distribution[note_length] = 0
            notes_lengths_distribution[note_length] += 1
        return notes_lengths_distribution
    
    def distance(self, value_1, value_2):
        for k in value_1.keys():
            if k not in value_2:
                value_2[k] = 0
        
        for k in value_2.keys():
            if k not in value_1:
                value_1[k] = 0

        data_1 = []
        data_2 = []

        for k in value_1.keys():
            data_1.append(value_1[k])
            data_2.append(value_2[k])

        data_1 = np.array(data_1, dtype='float32')
        data_2 = np.array(data_2, dtype='float32')

        data_1 = data_1 / data_1.sum() + 1e-6
        data_2 = data_2 / data_2.sum() + 1e-6

        return kl_divergence(data_1, data_2)