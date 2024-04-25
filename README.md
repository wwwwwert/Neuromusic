# Neuromusic project 

## Project description

This project employs multiple LLM models to generate symbolic music. There is an end-to-end pipeline to train, evaluate and generate music with various models. The pipeline supports extensive logging information with WandB.

Models:
- Music Transformer
- ...

Tokenisers:
- REMI
- ...

Datasets:
- Maestro Dataset 2.0.0
- Los Angeles MIDI Dataset 3.1
- ...

## Project structure
- **/scripts** - project scripts
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test

## Installation guide

It is strongly recommended to use new virtual environment for this project. Project was developed with Python3.9 and Ubuntu 22.04.2 LTS.

To install all required dependencies and final model run:
```shell
./install_dependencies.sh
```

## Reproduce results
To run train _Music Transformer_ with _REMI_ tokenizer and _Los Angeles MIDI_ dataset:
```shell
python -m train -c scripts/configs/REMI/train_music_tranformer.json
```

To run test inference with _Los Angeles MIDI_ dataset with 512 prompt tokens and generate 512 tokens:
```
python test.py \
   -c scripts/configs/test_LAMD.json \
   -r best_model/model_best.pth \
   -o test_results_LAMD \
   --prompt_length 512 \
   --continue_length 512 \
   -b 1
```

To test model on a custom dataset you need to put MIDI files in some directory.
To run test with custom dataset in _custom_dataset_ directory:
```
python test.py \
   -c scripts/configs/test_custom.json \
   -r best_model/model_best.pth \
   -o test_results_custom \
   --prompt_length 512 \
   --continue_length 512 \
   -b 1 \
   -t custom_dataset/
```

## Inference evaluation
#### Quality Assessment Procedure
To evaluate quality of generated compositions the following metrics are proposed:
1. Pitch Class Distribution - distribution of used notes pitches
2. Notes Duration Distribution - distribution of duration of used notes
3. Harmonic Reduction - evaluated harmony reduction sequence
The eva

The evaluation script calculates the features of the prompt and the continuations of the original and generated compositions. It then calculates the difference between the features of the prompt and the continuations, resulting in two distributions of feature differences. An A/B test is performed on these distributions, and the p-value is stored. The Kolmogorov-Smirnov test is used for this. Histograms of distances distributions saved as well.

The script considers the KL divergence between the distributions of the first two features. For the third feature, the WordVec model was trained on the harmonic series from the test dataset. Embeddings were then calculated for the harmonic series of the prompt and continuation using the trained model, and cosine similarity was calculated. 

To train Word2Vec for harmony reduction with _Los Angeles MIDI_ dataset:
```
python train_word2vec.py \
   -c scripts/configs/train_word2vec.json \
   -o saved/word2vec.model
```

To evaluate the p-values of the proposed features on the generation results of test.py: 
```
python evaluate.py \
   -r test_results/results.json \
   -m saved/word2vec.model \
   -o evaluation_results
```

## Author
Dmitrii Uspenskii HSE AMI 4th year.
