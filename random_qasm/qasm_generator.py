import math
from enum import Enum
from dataclasses import dataclass
import random
import pandas as pd
import numpy as np
from scmr_embed.embedding_types import QASM, Gate, GateType, Layer
from scmr_embed.feature_extractor import FeatureExtractor


class QasmGenerator:
    def __init__(
        self, features: pd.DataFrame, num_qubits: int, num_layers: int
    ) -> None:
        self.features = features
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def save_qasm(self, qasm: QASM, path: str):
        qasm_lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.num_qubits}];",
        ]
        t_gate_line = "t q[{}];"
        cx_gate_line = "cx q[{}], q[{}];"
        for i, layer in enumerate(qasm.layers):
            qasm_lines.append(f"// Layer {i}")
            for gate in layer:
                if gate.type is GateType.CX:
                    qasm_lines.append(cx_gate_line.format(gate.data[0], gate.data[1]))
                elif gate.type is GateType.T:
                    qasm_lines.append(t_gate_line.format(gate.data[0]))
        with open(path, "w") as f:
            f.write("\n".join(qasm_lines))

    def sample_features(self, feature: str, num_samples: int):
        """
        Gausian Mixture Model Sampling
        """
        means = np.array(self.features[feature + ".mean"])
        stds = np.array(self.features[feature + ".std"])
        assert len(means) == len(stds)
        # Randomly choose components with equal probability
        components = np.random.choice(len(means), size=num_samples)
        # Generate samples
        return np.random.normal(loc=means[components], scale=stds[components])

    def construct_perfect_parallel_qasm(self, num_layers: int) -> QASM:
        qubits = set(range(self.num_qubits))
        layers: list[Layer] = []
        for i in range(num_layers):
            layer: list[Gate] = []
            for q in range(self.num_qubits):
                layer.append(Gate(type=GateType.T, data=(q, -1)))
            layers.append(layer)
        return QASM(layers=layers, qubits=qubits)

    def construct_parallel(self, parallel: float) -> QASM:
        gate_layer_ratio = (self.num_qubits * parallel).as_integer_ratio()
        print(f"Gate:Layer = {gate_layer_ratio}")
        necessary_padding = math.ceil(5 / (gate_layer_ratio[0] - gate_layer_ratio[1]))
        gate_layer_ratio = (
            gate_layer_ratio[0] * necessary_padding,
            gate_layer_ratio[1] * necessary_padding,
        )
        print(f"Gate:Layer post padding = {gate_layer_ratio}")
        qasm = self.construct_perfect_parallel_qasm(gate_layer_ratio[1])

        # One qubit needs to be maintained
        persistent_qubit = random.randrange(self.num_qubits)

        layer_track = len(qasm.layers) - 1
        num_remove_gates = (gate_layer_ratio[1] * self.num_qubits) - gate_layer_ratio[0]
        for i in range(num_remove_gates):
            if len(qasm.layers[layer_track]) == 1:
                layer_track -= 1
            candidate_removals = [
                i
                for i, gate in enumerate(qasm.layers[layer_track])
                if persistent_qubit not in gate.data
            ]
            qasm.layers[layer_track].pop(random.choice(candidate_removals))

        return qasm

    def arrange_cx_ratio(self, cx_ratio: float, precision: float, limit: int = 1000):
        @dataclass
        class OP:
            class OP_TYPE(Enum):
                add_layer = 0
                add_cx = 1
                remove_t = 2

            t: OP_TYPE
            i: int

        cx_gate_ratios = [(0, self.num_qubits)]
        cur_ratio = 0

        i = 0
        while abs(cur_ratio - cx_ratio) > precision and i < limit:
            i += 1
            best_ratio = cur_ratio * (len(cx_gate_ratios) / (len(cx_gate_ratios) + 1))
            best_diff = abs(best_ratio - cx_ratio)
            best_op = OP(t=OP.OP_TYPE.add_layer, i=0)
            for i, (cx, count) in enumerate(cx_gate_ratios):
                if cx < count - 1:
                    add_cx_ratio = cur_ratio + (
                        (cx + 1) / (count - 1) - cx / count
                    ) / len(cx_gate_ratios)

                    add_cx_diff = abs(add_cx_ratio - cx_ratio)
                    if add_cx_diff < best_diff:
                        best_diff = add_cx_diff
                        best_ratio = add_cx_ratio
                        best_op = OP(t=OP.OP_TYPE.add_cx, i=i)

                if cx == count - 1:
                    remove_t_ratio = cur_ratio + (cx / (count - 1) - cx / count) / len(
                        cx_gate_ratios
                    )

                    remove_t_diff = abs(remove_t_ratio - cx_ratio)
                    if remove_t_diff < best_diff:
                        best_diff = remove_t_diff
                        best_ratio = remove_t_ratio
                        best_op = OP(t=OP.OP_TYPE.remove_t, i=i)

            # print(cur_ratio, best_ratio, best_op)

            cur_ratio = best_ratio
            if best_op.t is OP.OP_TYPE.add_layer:
                cx_gate_ratios.append((0, self.num_qubits))
            elif best_op.t is OP.OP_TYPE.add_cx:
                layer = cx_gate_ratios[best_op.i]
                cx_gate_ratios[best_op.i] = (layer[0] + 1, layer[1] - 1)
            elif best_op.t is OP.OP_TYPE.remove_t:
                layer = cx_gate_ratios[best_op.i]
                cx_gate_ratios[best_op.i] = (layer[0], layer[1] - 1)

        qasm = QASM(layers=[], qubits=set(range(self.num_qubits)))
        missing = random.randrange(self.num_qubits)
        for cxs, gates in cx_gate_ratios:
            layer = []
            available_qubits = set(range(self.num_qubits))
            cx_remaining = cxs
            new_missing = missing
            # If a missing is required, remove it from the available qubits
            if cxs == gates and self.num_qubits % 2 == 1:
                new_missing = (
                    missing + random.randrange(self.num_qubits - 1)
                ) % self.num_qubits
                available_qubits.remove(new_missing)
            for i in range(gates):
                # Place ts if all of the cxs have been placed
                if cx_remaining == 0:
                    layer.append(
                        Gate(type=GateType.T, data=(available_qubits.pop(), -1))
                    )
                    continue
                cx_remaining -= 1
                # Try to connect the previous missing if able
                if missing in available_qubits:
                    available_qubits.remove(missing)
                    layer.append(
                        Gate(type=GateType.CX, data=(missing, available_qubits.pop()))
                    )
                    continue
                # Connect cx gates
                layer.append(
                    Gate(
                        type=GateType.CX,
                        data=(available_qubits.pop(), available_qubits.pop()),
                    )
                )

            missing = new_missing
            qasm.layers.append(layer)
        return qasm

    def keep_going_cx(self, cx_mean: float, cx_std: float):
        # ratios =
        pass
