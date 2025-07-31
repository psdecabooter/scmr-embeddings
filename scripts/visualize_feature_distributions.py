import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="A script to visualize embedding csv files"
    )
    parser.add_argument("path", help="Path to the directory of embeddings csvs")
    args = parser.parse_args()
    if not os.path.exists(args.path) or not os.path.isdir(args.path):
        print(f"Expected a directory at {args.path}")
        exit(1)

    csv_names: list[str] = list(
        filter(lambda name: name.endswith(".csv"), os.listdir(args.path))
    )
    csv_paths: list[str] = [os.path.join(args.path, csv_name) for csv_name in csv_names]
    print(f"Found files: {', '.join(csv_names)}")

    for id, path in enumerate(csv_paths):
        # read file
        df = pd.read_csv(path)
        median_labels = list(filter(lambda label: label.endswith("median"), df.columns))
        mean_labels = list(filter(lambda label: label.endswith("mean"), df.columns))
        std_labels = list(filter(lambda label: label.endswith("std"), df.columns))
        num_features = len(mean_labels)

        # Plot# Determine grid size
        cols = math.ceil(math.sqrt(num_features))
        rows = math.ceil(num_features / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))
        axs = axs.flatten()
        fig.suptitle("QASM Features: " + csv_names[id])
        # Single csv case
        if len(csv_paths) == 1:
            axs = [axs]

        for i in range(num_features):
            axs[i].set_title(std_labels[i][:-4])

            median_data = list(df[median_labels[i]])
            mean_data = list(df[mean_labels[i]])
            std_data = list(df[std_labels[i]])
            data = sorted(zip(mean_data, std_data, median_data), key=lambda x: x[0])

            x_index = np.arange(len(data))

            axs[i].scatter(
                x_index,
                [d[2] for d in data],
                color="red",
                marker="s",
                s=50,
                label="Median",
            )
            axs[i].errorbar(
                x_index,
                [d[0] for d in data],
                yerr=[d[1] for d in data],
                fmt="o",
                capsize=5,
                capthick=2,
                label="Mean ± STD",
            )
            axs[i].set_ylim(0, 1)  # Y-axis from 0 to 1

        # Turn off any unused axes
        for i in range(len(csv_paths), len(axs)):
            fig.delaxes(axs[i])

        plt.subplots_adjust(
            top=0.9,  # suptitle space
            bottom=0.05,
            left=0.05,
            right=0.95,
            hspace=0.6,  # more vertical space
            wspace=0.6,  # more horizontal space
        )
        if not os.path.exists(os.path.join(args.path, "images")):
            os.mkdir(os.path.join(args.path, "images"))
        plt.savefig(os.path.join(args.path, "images", csv_names[id][:-4] + ".png"))
        plt.clf()

    # plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    # Adjust layout spacing and room for suptitle
    # plt.show()
    # plt.tight_layout()


if __name__ == "__main__":
    main()
