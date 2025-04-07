#!/bin/bash

# Run PyFitSeq for Nlim replicate 1 segment 0
python3 /Users/dawsontrotman/Documents/GitHub/PyFitSeq/PyFitSeq/pyfitseq.py \
  -i /Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/fluctuating_env/pyfitseq_input_Nlim_rep1.csv \
  -t 0 24 48 72 96 120 144 168 240 \
  -o /Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/fluctuating_env/results_Nlim_rep1
