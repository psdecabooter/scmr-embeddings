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
    parser.add_argument("-o", "--output", help="Path to the output png")
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
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 7, rows * 7), subplot_kw=dict(polar=True)
    )
    # Single csv case
    if len(csv_paths) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    fig.suptitle("QASM Features")
    for i, path in enumerate(csv_paths):
        axs[i].set_title(csv_names[i])
        df = pd.read_csv(path)
        labels = list(filter(lambda label: label.endswith("mean"), df.columns))
        num_vars = len(labels)

        # data = {"means": [np.mean(df[label]) for label in labels]}
        data = {
            row["file_name"]: [row[label] for label in labels]
            for i, row in df.iterrows()
        }

        # Prepare the angles for the radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        for label, values in data.items():
            values += values[:1]  # Close the loop
            axs[i].plot(angles, values, label=label)
            axs[i].fill(angles, values, alpha=0.25)

        axs[i].set_xticks(angles[:-1])
        axs[i].set_xticklabels(labels)

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
    if args.path is not None:
        plt.savefig(args.output)
        return
    plt.show()
    # plt.tight_layout()


if __name__ == "__main__":
    main()
