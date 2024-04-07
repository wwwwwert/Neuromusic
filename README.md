# Neuromusic project 

## Project description

This project employs multiple LLM models to generate symbolic music. There is an end-to-end pipeline to train, evaluate and generate music with various models. The pipeline supports extensive logging information with WandB.

Training process report, notebook.

Models:
- Music Transformer
- ...

Tokenisers:
- REMI
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
To run train with _Los Angeles MIDI_ dataset:
```shell
python -m train -c scripts/configs/train_config.json
```

To run test inference with _Los Angeles MIDI_ dataset:
```
python test.py \
   -c scripts/configs/test_LAMD.json \
   -r best_model/model_best.pth \
   -o test_results_LAMD \
   -b 1
```

To test model on a custom dataset you need to put MIDI files in some directory.
To run test with custom dataset in _custom_dataset_ directory:
```
python test.py \
   -c scripts/configs/test_custom.json \
   -r best_model/model_best.pth \
   -o test_results_custom \
   -b 1 \
   -t custom_dataset/
```

## Author
Dmitrii Uspenskii HSE AMI 4th year.
