from .embedding_types import QASM, GateType, Features
import numpy as np


class FeatureExtractor:
    def __init__(self, qasm: QASM):
        self.qasm = qasm

    def cx_layer_ratio(self):
        ratios = []
        for layer in self.qasm.layers:
            cx = 0
            for gate in layer:
                cx += int(gate.type is GateType.CX)
            ratios.append(cx / len(layer))
        return np.array(ratios)

    def qubit_degrees(self):
        degrees: dict[int, set] = {qubit: set() for qubit in self.qasm.qubits}
        for layer in self.qasm.layers:
            for gate in layer:
                if gate.type is GateType.T:
                    continue
                degrees[gate.data[0]].add(gate.data[1])
                degrees[gate.data[1]].add(gate.data[0])
        return np.array([len(interacts) for interacts in list(degrees.values())])

    def parallelism(self):
        layer_gates = []
        for layer in self.qasm.layers:
            layer_gates.append(len(layer))
        return np.array(layer_gates)

    def cx_parallelism(self):
        cx_layer_gates = []
        for layer in self.qasm.layers:
            cx_layer_gates.append(
                len(list(filter(lambda qubit: qubit.type is GateType.CX, layer)))
            )
        return np.array(cx_layer_gates)

    def t_parallelism(self):
        t_layer_gates = []
        for layer in self.qasm.layers:
            t_layer_gates.append(
                len(list(filter(lambda qubit: qubit.type is GateType.T, layer)))
            )
        return np.array(t_layer_gates)

    def mean_time_between_t(self):
        time_between_t: dict[int, list[int]] = {}
        for layer in self.qasm.layers:
            t_qubits = set()
            for gate in layer:
                if gate.type is GateType.CX:
                    continue
                qubit = gate.data[0]
                if qubit not in time_between_t:
                    time_between_t[qubit] = []
                t_qubits.add(gate.data[0])
            for qubit in list(time_between_t.keys()):
                if qubit in t_qubits:
                    time_between_t[qubit].append(0)
                else:
                    time_between_t[qubit][-1] += 1
        return np.array(
            [
                sum(v[:-1]) / len(v[:-1])
                for _, v in list(time_between_t.items())
                if len(v) > 1
            ]
        )

    def all_features(self) -> dict[Features, np.ndarray]:
        # metrics
        cx_ratios = []
        qubit_degrees: dict[int, set] = {qubit: set() for qubit in self.qasm.qubits}
        layer_gates = []
        cx_layer_gates = []
        t_layer_gates = []
        time_between_t: dict[int, list[int]] = {}

        for layer in self.qasm.layers:
            # temp metrics
            cx_count = 0
            t_count = 0
            layer_gates.append(len(layer))
            t_qubits = set()

            for gate in layer:
                if gate.type is GateType.CX:
                    cx_count += 1
                    qubit_degrees[gate.data[0]].add(gate.data[1])
                    qubit_degrees[gate.data[1]].add(gate.data[0])
                elif gate.type is GateType.T:
                    t_count += 1
                    qubit = gate.data[0]
                    if qubit not in time_between_t:
                        time_between_t[qubit] = []
                    t_qubits.add(gate.data[0])
            cx_layer_gates.append(cx_count)
            t_layer_gates.append(t_count)
            cx_ratios.append(cx_count / len(layer))
            for qubit in list(time_between_t.keys()):
                if qubit in t_qubits:
                    time_between_t[qubit].append(0)
                else:
                    time_between_t[qubit][-1] += 1

        features = {
            Features.cx_layer_ratio: np.array(cx_ratios),
            Features.qubit_degrees: np.array(
                [len(interacts) for interacts in list(qubit_degrees.values())]
            ),
            Features.parallelism: np.array(layer_gates),
            Features.cx_parallelism: np.array(cx_layer_gates),
            Features.t_parallelism: np.array(t_layer_gates),
            Features.mean_time_between_t: np.array(
                [
                    sum(v[:-1]) / len(v[:-1])
                    for _, v in list(time_between_t.items())
                    if len(v) > 1
                ]
            ),
        }

        return features
