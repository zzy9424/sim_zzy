#!/bin/bash

cuda_ids=(0 1)
commands=(
  "python safepo/single_agent/ppo.py"
  "python safepo/single_agent/trpo.py"
)

for ((i=0; i<${#cuda_ids[@]}; i++)); do
  cuda_id=${cuda_ids[$i]}
  command=${commands[$i]}" --device=cuda:${cuda_id}"
  
  echo "Running command on CUDA device ${cuda_id}: ${command}"
  nohup ${command} > cuda${cuda_id}.out 2>&1 &
done