#!/usr/bin/env bash


####################  FedAvg ######################
aggregation_out="avg"

outf="outputs/outputs_femnist_logreg"
logf="outputs/log_femnist_logreg"
num_rounds=6000
batch_size=50
num_epochs=5
clients_per_round=100
lr=8e-2

main_args=" -dataset femnist -model log_reg -lr $lr  "
options=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} "



for seed in 1 2 3 4 5
do
    corruption="clean"
    time python models/main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${seed} 2>&1

    for corruption in p_x omniscient
    do
        for frac in 0.01 0.25
        do
            time python models/main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1
        done
    done
done


####################  RobustFedAvg  ######################
aggregation="geom_median"
aggregation_out="gm"

outf="outputs/outputs_femnist_logreg"
logf="outputs/log_femnist_logreg"
num_rounds=2000
batch_size=50
num_epochs=5
clients_per_round=100
lr=8e-2
niter=2

main_args=" -dataset femnist -model log_reg -lr $lr --aggregation $aggregation "
options=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} --weiszfeld-maxiter ${niter} "


for seed in 1 2 3 4 5
do
    corruption="clean"
    time python models/main.py ${main_args} $options  --seed $seed --output_summary_file ${outf}_${aggregation_out}_${corruption}_${niter}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${niter}_${seed} 2>&1

    for corruption in p_x omniscient
    do
        for frac in 0.01 0.25
        do
            time python models/main.py ${main_args} $options  --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${niter}_${seed}  > ${logf}_${aggregation_out}_${corruption}_${frac}_${niter}_${seed} 2>&1
        done
    done
done


#################### Minibatch SGD ######################
outf="outputs/outputs_femnist_logreg_mb"
logf="outputs/log_femnist_logreg_mb"
num_rounds=6000
minibatch=0.1
clients_per_round=100
eval_every=50
lr=8e-2
lrd=1

main_args="-dataset femnist -model log_reg"
common=" --num-rounds ${num_rounds} --eval-every ${eval_every} --clients-per-round ${clients_per_round} --minibatch $minibatch -lr $lr --lr-decay $lrd  "


for seed in 1 2 3 4 5
do

	corruption=clean
	time python models/main.py ${main_args}  $common --seed $seed --output_summary_file ${outf}_${corruption}_${seed} > ${logf}_${corruption}_${seed} 2>&1

	for corruption in p_x omniscient
	do
        for frac in 0.01 0.25
        do
            time python models/main.py ${main_args}  $common --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${seed}  > ${logf}_${corruption}_${frac}_${seed} 2>&1
        done
	done


done
