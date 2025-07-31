import random
import os
import argparse
import pandas as pd
import numpy as np


def sample_features(feature: str, num_samples: int, df):
    """
    Gausian Mixture Model Sampling
    """
    means = np.array(df[feature + ".mean"])
    stds = np.array(df[feature + ".std"])
    assert len(means) == len(stds)
    # Randomly choose components with equal probability
    components = np.random.choice(len(means), size=num_samples)
    # Generate samples
    return np.random.normal(loc=means[components], scale=stds[components])


def main():
    parser = argparse.ArgumentParser(
        description="Radom QASM circuit generator according to features"
    )
    parser.add_argument("path", help="Path to feature csv")
    parser.add_argument("gates", help="The number of gates in the output circuit")
    parser.add_argument("qubits", help="The number of qubits in the output circuit")
    parser.add_argument("-o", "--output", help="Name of the output path")
    args = parser.parse_args()
    if not os.path.exists(args.path) or not str(args.path).endswith(".csv"):
        print(f"Could not find csv at {args.path}")
        exit(1)

    features = pd.read_csv(args.path)
    feature_names = [col[:-5] for col in features.columns if col.endswith("mean")]


    circuit_length = int(args.gates)
    circuit_qubits = int(args.qubits)
    output_path = "out.qasm" if args.output is None else args.output

    qasm_lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit_qubits}];",
    ]
    t_gate_line = "t q[{}];"
    cx_gate_line = "cx q[{}], q[{}];"
    for i in range(circuit_length):
        choice = bool(random.randrange(2))
        if choice:
            qasm_lines.append(t_gate_line.format(random.randrange(circuit_qubits)))
        else:
            qasm_lines.append(
                cx_gate_line.format(
                    random.randrange(circuit_qubits), random.randrange(circuit_qubits)
                )
            )

    with open(output_path, "w") as f:
        f.write("\n".join(qasm_lines))


if __name__ == "__main__":
    main()
