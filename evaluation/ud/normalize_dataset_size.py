import json
import logging
import numpy as np
import os
import pandas as pd
import random

from argparse import ArgumentParser
from conllu import parse
from pathlib import Path

LANGS = {
    "primary": ["en", "he", "zh", "vi", "ko", "tr", "el"],
    "secondary": ["id", "fr", "de", "tl", "ru", "ja", "th"],
    "tertiary": ["my", "hi", "ka", "fi", "es", "fa"],
}


def seed_everything(seed_value=42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)


def get_filepaths(args, language: str) -> dict[str]:
    language_treebank_mapping = json.load(open(args.treebank_mapping, "r"))
    treebank = language_treebank_mapping[language]
    if treebank is None:
        raise ValueError(f"Treebank not found for {language}")
    treebank_path = f"{args.ud_treebanks_dir}/{treebank}"

    # find train, dev, test filenames
    train, dev, test = None, None, None
    for filename in Path(treebank_path).rglob("*"):
        if "dev" in filename.name and ".conllu" in filename.suffixes:
            dev = filename
        elif "train" in filename.name and ".conllu" in filename.suffixes:
            train = filename
        elif "test" in filename.name and ".conllu" in filename.suffixes:
            test = filename

    return {"train": str(train), "dev": str(dev), "test": str(test)}


def load_datasets(file_paths_dict: dict[str]) -> dict:
    datasets = {}
    for l, filepaths in file_paths_dict.items():
        logging.info(f"Parsing {l} treebanks...")
        temp = {}
        sizes = []
        for kind, path in filepaths.items():
            _temp = parse(open(path).read())
            random.shuffle(_temp)
            temp[kind] = _temp
            sizes.append(sum(len(elem) for elem in _temp))
        temp["sizes"] = sizes
        datasets[l] = temp
    return datasets


def find_smallest_dataset(datasets: dict[str:dict]) -> dict:
    size_list = np.array([datasets[l]["sizes"] for l in datasets.keys()])

    min_values = np.min(size_list, axis=0)

    return {"train": min_values[0], "dev": min_values[1], "test": min_values[2]}


def normalize_dataset(dataset: dict[str:dict], max_tokens: int) -> list:
    subset = []
    counter = 0
    for sentence in dataset:
        subset.append(sentence)
        counter += len(sentence)
        if counter >= max_tokens:
            break
    return subset


def generate_statistics(datasets: dict[str:dict]) -> None:
    df = pd.DataFrame(
        columns=[
            "language",
            "n_train_sents",
            "n_train_tokens",
            "n_dev_sents",
            "n_dev_tokens",
            "n_test_sents",
            "n_test_tokens",
        ]
    )
    for i, (l, data) in enumerate(datasets.items()):
        df.loc[i] = (
            l,
            len(data["train"]),
            sum(len(elem) for elem in data["train"]),
            len(data["dev"]),
            sum(len(elem) for elem in data["dev"]),
            len(data["test"]),
            sum(len(elem) for elem in data["test"]),
        )

    n_tkns = sum([df["n_train_tokens"], df["n_dev_tokens"], df["n_test_tokens"]])
    n_sents = sum([df["n_train_sents"], df["n_dev_sents"], df["n_test_sents"]])

    df["avg_tokens_per_sentence"] = n_tkns / n_sents
    logging.info(str(df))


def save_subsets(
    args, datasets: dict[str:dict], file_paths_dict: dict[str : dict[str]]
):
    def _save_subset():
        # removes .conllu ending
        original_filename = file_paths_dict[l][splt].split("/")[-1][:-7]
        subset_filepath = lang_dir / (original_filename + "_subset.conllu")
        with open(subset_filepath, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence.serialize())

    args.out_dir = Path(args.out_dir)
    if not args.out_dir.exists():
        args.out_dir.mkdir()

    for l, data in datasets.items():
        lang_dir = args.out_dir / l
        if not lang_dir.exists():
            lang_dir.mkdir()
        for splt, sentences in data.items():
            if splt != "sizes":
                _save_subset()


def main(args) -> None:
    # Locate treebanks
    file_paths = {}
    for l in args.languages:
        file_paths[l] = get_filepaths(args, l)

    datasets = load_datasets(file_paths)

    normalized_sizes = find_smallest_dataset(datasets)
    logging.info(f"Sizes of the normalized datasets: {normalized_sizes}")

    logging.info("Normalizing datasets...")
    for l, data in datasets.items():
        for splt, sentences in data.items():
            if splt != "sizes":
                datasets[l][splt] = normalize_dataset(sentences, normalized_sizes[splt])

    logging.info("Generating statistics...")
    generate_statistics(datasets)

    logging.info("Saving normalized subsets...")
    save_subsets(args, datasets, file_paths)


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="normalize_dataset_size.log",
        filemode="w",
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--language_set",
        "-l",
        help="choose language set: primary, secondary, or tertiary",
        choices=["primary", "secondary", "tertiary"],
        default="primary",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        help="Path to the output directory",
        default="subsets/",
    )
    parser.add_argument(
        "--treebank_mapping",
        help="Filepath to mapping of language to UD treebank",
        default="language_treebank_mapping.json",
    )
    parser.add_argument(
        "--ud_treebanks_dir",
        help="Path to the UD treebanks directory",
        default="ud-treebanks-v2.14",
    )
    args = parser.parse_args()

    args.languages = LANGS[args.language_set]

    logging.info("ARGUMENTS")
    for k, v in args.__dict__.items():
        logging.info(f"{k}: {v}")

    seed_everything()
    main(args)
