#!/bin/bash

# Default values
MODEL=${1:-"llama-7b-hf"}
DATASET=${2:-"coqa"}
DEVICE=${3:-"cuda:0"}
FRACTION=${4:-"0.1"}

# Run the pipeline
python -m pipeline.generate --model $MODEL --dataset $DATASET --device $DEVICE --fraction_of_data_to_use $FRACTION

echo "Completed running $MODEL on $DATASET ($FRACTION of data) using device $DEVICE"
