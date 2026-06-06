OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];

x q[2];

measure q -> c;
