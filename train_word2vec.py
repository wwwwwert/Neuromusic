import argparse
import json
import os
import warnings
from functools import partial
from multiprocessing import Pool

import gensim
from miditok import REMI, TokenizerConfig
from tqdm import tqdm

import scripts.datasets
from scripts.base.base_dataset import BaseDataset
from scripts.base.base_feature import FeatureBase
from scripts.evaluation import HarmonicReductionFeature
from scripts.evaluation.utils import open_midi
from scripts.utils.parse_config import ConfigParser


def main(dataset: BaseDataset, model_args, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    args = [
        [dataset._index[i]['midi_path'], HarmonicReductionFeature()]
        for i in range(len(dataset))
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Pool(8) as p:
            harmony_seqs = list(tqdm(
                p.imap(calc_feature, args), 
                total=len(dataset)
            ))
    harmony_seqs = [seq for seq in harmony_seqs if seq is not None]
    
    with open('saved/kek.json', 'w') as fp:
        json.dump({'harmony_seqs': harmony_seqs}, fp, indent=2)

    model = gensim.models.Word2Vec(
        harmony_seqs, 
        **model_args
    )

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)

def calc_feature(args):
    try:
        midi_path, feature = args
        feature = HarmonicReductionFeature()
        m21_score = open_midi(midi_path, remove_drums=True)
        return feature(m21_score)
    except:
        return


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train Word2Vec for harmonic reduction")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path",
    )
    args.add_argument(
        "-o",
        "--output",
        default="saved/word2vec.model",
        type=str,
        help="Path to save model",
    )

    args = args.parse_args()

    with open(args.config, 'r') as fp:
        configs = json.load(fp)

    tokenizer_config = TokenizerConfig(
        num_velocities=16, 
        use_chords=True, 
        use_programs=True, 
        remove_duplicated_notes=True, 
        delete_equal_successive_tempo_changes=True,
        delete_equal_successive_time_sig_changes=True
    )
    tokenizer = REMI(tokenizer_config)

    with open('scripts/configs/train.json') as fp:
        config_parser = ConfigParser(json.load(fp))

    dataset = ConfigParser.init_obj(
        configs['dataset'], 
        scripts.datasets, 
        midi_encoder=tokenizer, 
        config_parser=config_parser
    )

    main(dataset, configs['Word2Vec_args'], args.output)