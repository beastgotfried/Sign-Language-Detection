import argparse
import os
import pickle

from phase2_features import extract_features

DATA_DIR = "./data"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hand features from collected landmark data.")
    parser.add_argument("--input-pickle", default=os.path.join(DATA_DIR, "collected_data.pickle"))
    parser.add_argument("--output-pickle", default=os.path.join(DATA_DIR, "processed_data.pickle"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_pickle = args.input_pickle
    output_pickle = args.output_pickle

    print(f"Loading the raw data from {input_pickle}")
    with open(input_pickle, "rb") as f:
        dataset = pickle.load(f)

    raw_data = dataset["data"]
    labels = dataset["labels"]
    processed_data = []

    print("extracting features now")
    for feature in raw_data:
        extracted_vector = extract_features(feature)
        processed_data.append(extracted_vector.tolist())

    output_data = {"data": processed_data, "labels": labels}

    with open(output_pickle, "wb") as f:
        pickle.dump(output_data, f)

    print("Extraction complete")


if __name__ == "__main__":
    main()

