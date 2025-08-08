from random_qasm.qasm_generator import QasmGenerator
import pandas as pd

gen = QasmGenerator(pd.DataFrame(), 4, 10)
# qasm = gen.construct_parallel(0.375)
# qasm = gen.arrange_cx_ratio(0.675, 0.0001, 10_000)
ratio_layers = gen.cx_ratio_and_parallel(0.675, 0.375, 0.0001, 10_000)
# qasm = gen.from_ordered_layers(ratio_layers)
# qasm = gen.from_degree_layers(ratio_layers, 5 / 6)
# qasm = gen.loop_from_degree_layers(ratio_layers, 5 / 6)
qasm = gen.loop_from_degree_layers(ratio_layers, 5 / 6)
gen.validate_qasm(qasm)
gen.save_qasm(qasm, "out.qasm")
