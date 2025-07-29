OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
t q[4]; // layer 0
cx q[0], q[2]; // layer 0
t q[3]; // layer 0
t q[1]; // layer 0
t q[2]; // layer 1
cx q[1], q[0]; // layer 1
t q[1]; // layer 2
t q[1]; // layer 3
t q[0]; // layer 2
cx q[3], q[2]; // layer 2
cx q[3], q[1]; // layer 4
t q[3]; // layer 5
cx q[0], q[2]; // layer 3
t q[1]; // layer 5
t q[0]; // layer 4
cx q[3], q[0]; // layer 6
t q[2]; // layer 4
cx q[3], q[2]; // layer 7
t q[0]; // layer 7
cx q[1], q[3]; // layer 8
cx q[2], q[0]; // layer 8
t q[3]; // layer 9
t q[2]; // layer 9
t q[1]; // layer 9
t q[0]; // layer 9
cx q[2], q[0]; // layer 10