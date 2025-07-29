OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[0], q[2]; // layer 0
t q[1]; // layer 0
t q[2]; // layer 1
cx q[1], q[0]; // layer 1
t q[1]; // layer 2
t q[1]; // layer 3
t q[0]; // layer 2
cx q[3], q[2]; // layer 2
cx q[3], q[1]; // layer 4
cx q[0], q[2]; // layer 3
t q[1]; // layer 5
cx q[3], q[0]; // layer 5
