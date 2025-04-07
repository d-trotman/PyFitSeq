#!/bin/bash

# Master script to run all PyFitSeq analyses

echo 'Running Clim replicate 1...'
./run_pyfitseq_Clim_rep1.sh
echo 'Done!'

echo 'Running Clim replicate 2...'
./run_pyfitseq_Clim_rep2.sh
echo 'Done!'

echo 'Running Clim replicate 3...'
./run_pyfitseq_Clim_rep3.sh
echo 'Done!'

echo 'Running Nlim replicate 1...'
./run_pyfitseq_Nlim_rep1.sh
echo 'Done!'

echo 'Running Nlim replicate 3...'
./run_pyfitseq_Nlim_rep3.sh
echo 'Done!'

echo 'Running Nlim replicate 2...'
./run_pyfitseq_Nlim_rep2.sh
echo 'Done!'

echo 'Running PulseAS replicate 1...'
./run_pyfitseq_PulseAS_rep1.sh
echo 'Done!'

echo 'Running PulseAS replicate 2...'
./run_pyfitseq_PulseAS_rep2.sh
echo 'Done!'

echo 'Running PulseAS replicate 3...'
./run_pyfitseq_PulseAS_rep3.sh
echo 'Done!'

echo 'Running PulseGln replicate 1...'
./run_pyfitseq_PulseGln_rep1.sh
echo 'Done!'

echo 'Running PulseGln replicate 2...'
./run_pyfitseq_PulseGln_rep2.sh
echo 'Done!'

echo 'Running PulseGln replicate 3...'
./run_pyfitseq_PulseGln_rep3.sh
echo 'Done!'

echo 'Running Switch replicate 1...'
./run_pyfitseq_Switch_rep1.sh
echo 'Done!'

echo 'Running Switch replicate 2...'
./run_pyfitseq_Switch_rep2.sh
echo 'Done!'

echo 'Running Switch replicate 3...'
./run_pyfitseq_Switch_rep3.sh
echo 'Done!'

