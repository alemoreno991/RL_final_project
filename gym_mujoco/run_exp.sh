#!/bin/bash
time=`date +%Y-%m-%d_%T`

#!/bin/bash
for i in {0..10}
do
    python main_SAC.py --run $i --time $time
done