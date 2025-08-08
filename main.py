import argparse
import os
import pandas as pd
import numpy as np
from scmr_embed.embedding_types import parse_qasm
from scmr_embed.feature_extractor import FeatureExtractor
from random_qasm.qasm_generator import QasmGenerator


def main():
    parser = argparse.ArgumentParser(
        description="A script to extract features from qasm files"
    )
    parser.add_argument(
        "path", help="Path to the qasm circuit or qasm circuit directory"
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        help="Enable this flag if you want to extract features from qasm circuits in a directory",
    )
    parser.add_argument("-o", "--output", help="Set the name of the output csv")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Path {args.path} is not a valid path.")
        exit(1)

    qasm_file_paths: list[str] = []
    qasm_file_names: list[str] = []
    if args.dir:
        if not os.path.isdir(args.path):
            print(f"Expected a directory at {args.path}.")
            exit(1)
        qasm_file_paths.extend(
            [
                os.path.join(args.path, f)
                for f in filter(
                    lambda p: str(p).endswith(".qasm"), os.listdir(args.path)
                )
            ]
        )
        qasm_file_names.extend(
            filter(lambda p: str(p).endswith(".qasm"), os.listdir(args.path))
        )
    else:
        if os.path.isdir(args.path) or not str(args.path).endswith(".qasm"):
            print(f"Expected a qasm file at {args.path}")
            exit(1)
        qasm_file_paths.append(args.path)
        qasm_file_names.append(args.path)

    output_df = None
    for file_path, name in zip(qasm_file_paths, qasm_file_names):
        qasm = parse_qasm(file_path)
        # qasm_gen = QasmGenerator(pd.DataFrame(),1,1)
        # qasm_gen.save_qasm(qasm, path="out.qasm")
        extract_features = FeatureExtractor(qasm)
        features = extract_features.all_features()
        # print(features)
        features_df = pd.DataFrame()
        features_df["file_name"] = [name]
        for feature, data in features.items():
            if len(data) == 0:
                data = [0]
            features_df[f"{feature.value}.mean"] = [np.mean(data)]
            features_df[f"{feature.value}.median"] = [np.median(data)]
            features_df[f"{feature.value}.std"] = [np.std(data)]
        if output_df is None:
            output_df = features_df
        else:
            output_df = pd.concat([output_df, features_df], ignore_index=True)
    assert output_df is not None, "No qasm files found"
    output_df.to_csv("out.csv" if args.output is None else args.output, index=False)


if __name__ == "__main__":
    main()
