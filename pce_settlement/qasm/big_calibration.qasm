OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];

x q[3];

measure q -> c;
