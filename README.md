# Neuromusic project 

## Project description

This project employs multiple LLM models to generate symbolic music. There is an end-to-end pipeline to train, evaluate and generate music with various models. The pipeline supports extensive logging information with WandB. 

[WandB Report](https://wandb.ai/dauspenskiy_hse/diploma/reports/Neuromusic-Thesis---Vmlldzo3OTAyMDc1?accessToken=5u8lf27ayhbao9f0sh92aaok9x16k75pmrd59p9x6pu5rsii8tqhe1ldvirrjbyf) on training.

Models:
- Llama
- Music Transformer
- GPT-2

Tokenisers:
- REMI
- TSD
- Structured

Datasets:
- Maestro Dataset 2.0.0
- Los Angeles MIDI Dataset 3.1
- Custom datasets

The framework allows you to seamlessly integrate your own custom models, tokenizers, and datasets, providing you with greater flexibility and control over your project.

## Project structure
- **/scripts** - project scripts
- _evaluate.py_ - script to run generated compositions evaluation
- _install_dependencies.sh_ - script for dependencies installation and pre-trained models loading
- _requirements.txt_ - Python requirements list
- _test.py_ - script to run test
- _train.py_ - script to run train
- _train_word2vec.py_ - script to train Word2Vec model for further use in evaluation

## Installation guide

It is strongly recommended to use new virtual environment for this project. Project was developed with Python3.9, Ubuntu 22.04.2 LTS and CUDA 11.8.

To install all required dependencies and load pre-trained models run:
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
   --save_audio \ 
   -b 1
```

You can specify the number of elements in dataset by changing parameter _max_items_ in _test_LAMD.json_.

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

The evaluation script calculates the features of the prompt and the continuations of the original and generated compositions. It then calculates the difference between the features of the prompt and the continuations, resulting in two distributions of feature differences. The Kullback-Leibler divergence is employed to analyze these distributions. Histograms of distances distributions saved as well.

The script considers the KL divergence between the distributions of the first two features. For the third feature, the Word2Vec model was trained on the harmonic series from the test dataset. Embeddings were then calculated for the harmonic series of the prompt and continuation using the trained model, and cosine similarity was calculated. 

To train Word2Vec for harmony reduction with _Los Angeles MIDI_ dataset:
```
python train_word2vec.py \
   -c scripts/configs/train_word2vec.json \
   -o models/word2vec.model
```

To evaluate the KL divergence of the proposed features from results of test.py: 
```
python evaluate.py \
   -r test_results/results.json \
   -m models/word2vec.model \
   -o evaluation_results
```

## Author
Dmitrii Uspenskii HSE AMI 4th year.
dauspenskiy@edu.hse.ru
