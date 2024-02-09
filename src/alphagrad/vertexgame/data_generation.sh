#!/bin/bash
# $1 is prefix
# $2 is number of samples
# $3 is number of cores
# $4 is batchsize
# $5 is seed
# $6 generate scalar or mixed samples

for ((i=0;i<$3;i++))
do
    current_seed=$(($5*i))
    python3 ./generator.py --prefix $1 --seed $current_seed --num_samples $2 --scalar $6 --batchsize $4 &
done

