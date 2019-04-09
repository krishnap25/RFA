#!/usr/bin/env bash


# Effect of hyperparameter `--num_epochs` for number of passes in the local computation in each device

aggregation="geom_median"
aggregation_out="gm"

outf="outputs/fed/outputs_fem_logreg"
logf="outputs/fed/log_fem_logreg"
num_rounds=1700
batch_size=50
niter=2
lr=8e-2
clients_per_round=100

main_args=" -dataset femnist -model log_reg -lr $lr --aggregation $aggregation "


for seed in 1 2 3 4 5
do

    for num_epochs in 1 2 10 # note: 5 is already used in main plots
    do
        options=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} --weiszfeld-maxiter ${niter} "

        # clean
        corruption="clean"
        time python models/main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_nepochs${num_epochs}_${seed}  > ${logf}_${aggregation_out}_${corruption}_nepochs${num_epochs}_${seed} 2>&1

        for corruption in p_x omniscient
        do
            frac=0.25
            time python models/main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_nepochs${num_epochs}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_nepochs${num_epochs}_${seed} 2>&1
        done # corruption

    done # num_epochs

done # seed


