#!/bin/bash

# Run PyFitSeq for Nlim replicate 3 segment 0
python3 /Users/dawsontrotman/Documents/GitHub/PyFitSeq/PyFitSeq/pyfitseq.py \
  -i /Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/fluctuating_env/pyfitseq_input_Nlim_rep3.csv \
  -t 0 24 48 96 120 \
  -o /Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/fluctuating_env/results_Nlim_rep3
