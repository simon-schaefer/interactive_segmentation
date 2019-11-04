#!/bin/bash

#ssh sischaef@login.leonhard.ethz.ch
#cd /cluster/work/riner/users/PLR-2019/sischaef/interactive_segmentation

#bsub -W 8:00 -R "rusage[ngpus_excl_p=1]" < scripts/job.sh
#bsub -W 12:00 -n 8  < scripts/jobgrid.sh
