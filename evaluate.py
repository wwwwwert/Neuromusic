import argparse
import json
import os
from math import isinf, isnan
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from scripts.evaluation import (HarmonicReductionFeature,
                                NotesDurationDistributionFeature,
                                PitchClassDistributionFeature)
from scripts.evaluation.utils import open_midi
from scripts.utils import ROOT_PATH

plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['font.size'] = 12
sns.set_style('dark')


def main(results: dict, word2vec_path: str, output_path: Path):
    results_data = pd.DataFrame.from_dict(results)
    results_data['prompt_midi_path'] = results_data['composition_dir'].apply(lambda x: str(Path(x) / 'midi' / 'prompt.midi'))
    results_data['generated_midi_path'] = results_data['composition_dir'].apply(lambda x: str(Path(x) / 'midi' / 'generated.midi'))
    results_data['original_midi_path'] = results_data['composition_dir'].apply(lambda x: str(Path(x) / 'midi' / 'original.midi'))

    prompt_data = pd.DataFrame()
    prompt_data['midi_path'] = results_data['composition_dir'].apply(lambda x: str(Path(x) / 'midi' / 'prompt.midi'))
    generated_data = pd.DataFrame()
    generated_data['midi_path'] = results_data['composition_dir'].apply(lambda x: str(Path(x) / 'midi' / 'generated.midi'))
        
    features = [
        PitchClassDistributionFeature(), 
        HarmonicReductionFeature(model_path=word2vec_path),
        NotesDurationDistributionFeature(),
    ]

    prompt_to_generated_distances = {
        str(feature): []
        for feature in features
    }

    prompt_to_original_distances = {
        str(feature): []
        for feature in features
    }

    for idx, row in tqdm(results_data.iterrows(), total=results_data.shape[0]):
        prompt_m21 = open_midi(row['prompt_midi_path'], remove_drums=True)
        generated_m21 = open_midi(row['generated_midi_path'], remove_drums=True)
        original_m21 = open_midi(row['original_midi_path'], remove_drums=True)

        for feature in features:
            try:
                prompt_feature = feature(prompt_m21)
                generated_feature = feature(generated_m21)
                original_feature = feature(original_m21)

                dist_to_generated = feature.distance(prompt_feature, generated_feature)
                dist_to_original = feature.distance(prompt_feature, original_feature)

            except:
                continue

            if isnan(dist_to_generated) or isnan(dist_to_original):
                continue

            if isinf(dist_to_generated) or isinf(dist_to_original):
                continue

            prompt_to_generated_distances[str(feature)].append(dist_to_generated)
            prompt_to_original_distances[str(feature)].append(dist_to_original)

    pvalues = {}
    fig, axs = plt.subplots(1, len(features), figsize=(8 * len(features), 8))
    for idx, feature in enumerate(features):
        feature_name = str(feature)
        a = prompt_to_generated_distances[feature_name]
        b = prompt_to_original_distances[feature_name]
        sns.histplot(a, ax=axs[idx], bins=50, label='generated')
        sns.histplot(b, ax=axs[idx], bins=50, label='original')
        axs[idx].set_title(str(feature))
        _, pvalue = stats.kstest(a, b)
        pvalues[feature_name] = pvalue

    os.makedirs(output_path, exist_ok=True)
    plt.legend()
    plt.savefig(output_path / 'distances_distributions.png', bbox_inches='tight')
    with open(output_path / 'pvalues.json', 'w') as fp:
         json.dump(pvalues, fp, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test results of inference.")
    args.add_argument(
        "-r",
        "--results",
        default=None,
        type=str,
        help="Inference test results json file path",
    )

    args.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Path to save evaluation results",
    )

    args.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="Word2Vec model path for harmony reduction",
    )

    args = args.parse_args()

    with open(args.results, 'r') as fp:
            results = json.load(fp)

    main(results, args.model, Path(args.output))
