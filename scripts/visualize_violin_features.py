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

    # Plot# Determine grid size
    n = len(csv_paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))
    axs = axs.flatten()
    fig.suptitle("QASM Mean Features")
    # Single csv case
    if len(csv_paths) == 1:
        axs = [axs]
    for i, path in enumerate(csv_paths):
        ax = axs[i]
        ax.set_title(csv_names[i])
        df = pd.read_csv(path)
        labels = list(filter(lambda label: label.endswith("mean"), df.columns))

        # data = {"means": [np.mean(df[label]) for label in labels]}
        data = [df[label].values for label in labels]

        violin_parts = ax.violinplot(
            data, positions=range(1, len(data) + 1), showmeans=True, showmedians=True
        )

        # Color each violin differently
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))  # type: ignore
        for i, pc in enumerate(violin_parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    # Turn off any unused axes
    for i in range(len(csv_paths), len(axs)):
        fig.delaxes(axs[i])

    # plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    # Adjust layout spacing and room for suptitle
    plt.subplots_adjust(
        top=0.9,  # suptitle space
        bottom=0.05,
        left=0.05,
        right=0.95,
        hspace=0.6,  # more vertical space
        wspace=0.6,  # more horizontal space
    )
    plt.show()
    # plt.tight_layout()


if __name__ == "__main__":
    main()
