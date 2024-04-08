from music21.stream.base import Score as m21_score
from music21 import analysis, roman, stream, chord
from scripts.base.base_feature import FeatureBase


class DurationFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Duration Feature')

    def __call__(self, m21_score: m21_score):
        return m21_score.duration


class KeyFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Key Feature')

    def __call__(self, m21_score: m21_score):
        return m21_score.analyze('key')
    

class PitchDistributionFeature(FeatureBase):
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
        super().__init__('Pitch Distribution Feature')
    
    def __call__(self, m21_score: m21_score):
        pitch_count = analysis.pitchAnalysis.pitchAttributeCount(m21_score, 'name')
        res = []
        for pitch in self.pitches_range:
            if pitch in pitch_count:
                res.append(pitch_count[pitch])
            else:
                res.append(0)
        return res
    

class RhythmFeature(FeatureBase):
    def __init__(self) -> None:
        super().__init__('Rhythm Feature')

    def __call__(self, m21_score: m21_score):
        time_signatures = m21_score.getTimeSignatures()
        rhythms = []
        for ts in time_signatures:
            r = "{0}/{1}".format(ts.beatCount, ts.denominator)
            if not rhythms or rhythms[-1] != r:
                rhythms.append(r)
        return rhythms


class HarmonicReductionFeature(FeatureBase):
    def __init__(self, max_notes_per_chord: int=4):
        super().__init__('Harmonic Reduction Feature')

    def __call__(self, m21_score: m21_score):
        ret = []
        temp_midi = stream.Score()
        temp_midi_chords = m21_score.chordify()
        temp_midi.insert(0, temp_midi_chords)    
        music_key = temp_midi.analyze('key')
        max_notes_per_chord = 4   
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