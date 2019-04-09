#!/usr/bin/env bash


####################  FedAvg ######################

outf="outputs/outputs_shakespeare_avg"
logf="outputs/log_shakespeare_avg"
num_rounds=600
batch_size=8
num_epochs=1
clients_per_round=30
lr=0.64
lrd=2


main_args=" -dataset shakespeare -model stacked_lstm -lr $lr --lr-decay $lrd "
options=" ${main_args} --num-rounds ${num_rounds} --eval-every 20 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} "

for seed in 1 2 3 4 5
do
    time python models/main.py $options --seed $seed  --output_summary_file ${outf}_clean_${seed}  > ${logf}_clean_${seed} 2>&1
    for corruption in flip p_x
    do
        for frac in 0.1 0.25
        do
            time python models/main.py $options --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${seed}  > ${logf}_${corruption}_${frac}_${seed} 2>&1
        done
    done
done


####################  RobustFedAvg  ######################

outf="outputs/outputs_shakespeare_gm"
logf="outputs/log_shakespeare_gm"
num_rounds=200
batch_size=8
num_epochs=1
clients_per_round=30
lr=0.64
lrd=2
niter=2


main_args=" -dataset shakespeare -model stacked_lstm -lr $lr --lr-decay $lrd "
options=" ${main_args} --num-rounds ${num_rounds} --eval-every 20 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} --aggregation geom_median --weiszfeld-maxiter ${niter} "


for seed in 1 2 3 4 5
do
    time python models/main.py $options --seed $seed  --output_summary_file ${outf}_clean_${niter}_${seed}  > ${logf}_clean_${niter}_${seed} 2>&1
    for corruption in flip p_x
    do
        for frac in 0.1 0.25
        do
            time python models/main.py $options --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${seed}  > ${logf}_${corruption}_${frac}_${seed} 2>&1
        done
    done
done


#################### Minibatch SGD ######################
outf="outputs/outputs_shakespeare_mb"
logf="outputs/log_shakespeare_mb"
num_rounds=600
minibatch=0.1
eval_every=60
clients_per_round=30
lr=1.0
lrd=1

main_args=" -dataset shakespeare -model stacked_lstm "
common=" --num-rounds ${num_rounds} --eval-every ${eval_every} --clients-per-round ${clients_per_round} --minibatch $minibatch -lr $lr --lr-decay $lrd  "


for seed in 1 2 3 4 5
do
        corruption=clean
        time python models/main.py ${main_args}  $common --seed $seed --output_summary_file ${outf}_${corruption}_${seed} > ${logf}_${corruption}_${seed} 2>&1

        for corruption in flip omniscient
        do
            for frac in 0.01 0.25
            do
                time python models/main.py ${main_args}  $common --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${seed}  > ${logf}_${corruption}_${frac}_${seed} 2>&1
            done
        done
done
