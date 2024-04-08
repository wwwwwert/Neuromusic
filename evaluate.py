import argparse
import json
from typing import Union
from scripts.utils import ROOT_PATH

def main(results: Union[str, dict]):
    if results in None:
        raise ValueError('No results provided.')

    if isinstance(results, str):
        with open(results, 'r') as fp:
            results = json.load(fp)
    
    # получаем три датафрейма на признаки промпта, сгенерированных, оригинальных
    # считаем разницы между значениями признаков, получаем по два датафрейма с распределениями разниц
    # для каждого признака оцениваем статистическую значимость разницы
    ...

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test results of inference.")
    args.add_argument(
        "-r",
        "--results",
        default=None,
        type=str,
        help="results json file path",
    )

    main(args.results)
