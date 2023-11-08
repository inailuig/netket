#!/bin/bash
mpirun --hostfile /etc/hostfile -x NETKET_EXPERIMENTAL_SHARDING=1 -x NETKET_MPI_WARNING=0 pytest-3 -p no:warnings --color=yes --verbose -n 0 --tb=short test_sharding/test_sharding_distributed.py
