#!/usr/bin/env bash

# Effect of hyperparameter `--weiszfeld-maxiter`, the number of Weiszfeld iterations for geometric median computation
# in robust aggregation. Note that `n` Weiszfeld iterations will require `n+1` communication passes
# since the algorithm is started at the mean.

aggregation="geom_median"
aggregation_out="gm"

outf="outputs/fed/outputs_fem_logreg"
logf="outputs/fed/log_fem_logreg"
num_rounds=1700
batch_size=50
num_epochs=5
clients_per_round=100
lr=8e-2

main_args=" -dataset femnist -model log_reg -lr $lr --aggregation $aggregation "


for seed in 1 2 3 4 5
do
    for niter in 1 5 # note: 2 is already used in main plots
    do
        options=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} --weiszfeld-maxiter ${niter} "

        # clean
        corruption="clean"
        time python main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_${niter}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${niter}_${seed} 2>&1

        for corruption in p_x omniscient
        do
            frac=0.25
            time python main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${niter}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${niter}_${seed} 2>&1
        done # corruption
    done # niter
done # seed


echo "executing..."
f_ParallelExec $njobs "$cmds"
date

