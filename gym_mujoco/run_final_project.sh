#!/bin/bash
time=`date +%Y-%m-%d_%T`

NUM_RUNS=5
for (( i=0 ; i<$NUM_RUNS ; i++ )); 
do
    echo "**********************************"
    echo "**********************************"
    echo "**********************************"
    echo "Working on the $i/$NUM_RUNS run..." 
    echo "**********************************"
    echo "**********************************"
    echo "**********************************"
    python main_SAC.py  --run $i --suffix $time
    python main_DDPG.py --run $i --suffix $time
    python main_PPO.py  --run $i --suffix $time
done

echo "**********************************"
echo "Plotting..."
echo "**********************************" 
python analize_results.py --num_runs $NUM_RUNS --suffix $time