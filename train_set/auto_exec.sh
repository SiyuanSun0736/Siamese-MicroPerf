#!/bin/bash

# O1-g

./extract_elf.sh -v O1-g
sudo ./collect_dataset_testbench.sh -v O1-g

# O3-g
./extract_elf.sh -v O3-g
sudo ./collect_dataset_testbench.sh -v O3-g

# O2-bolt
./extract_elf.sh -v O2-bolt
sudo ./collect_dataset_testbench.sh -v O2-bolt

sudo ./bolt_profile.sh -v O2-bolt
sudo ./collect_dataset_testbench.sh -v O2-bolt-opt

# O3-bolt
./extract_elf.sh -v O3-bolt
sudo ./collect_dataset_testbench.sh -v O3-bolt

sudo ./bolt_profile.sh -v O3-bolt
sudo ./collect_dataset_testbench.sh -v O3-bolt-opt