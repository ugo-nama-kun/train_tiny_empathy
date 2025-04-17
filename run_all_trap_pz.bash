#!/bin/bash

python train_ippo_lstm_trap.py --track --seed=1 --wandb-group=no &
python train_ippo_lstm_trap.py --track --seed=1 --wandb-group=cognitive &
python train_ippo_lstm_trap.py --track --seed=1 --wandb-group=affective &
python train_ippo_lstm_trap.py --track --seed=1 --wandb-group=full &

python train_ippo_lstm_trap.py --track --seed=2 --wandb-group=no &
python train_ippo_lstm_trap.py --track --seed=2 --wandb-group=cognitive &
python train_ippo_lstm_trap.py --track --seed=2 --wandb-group=affective &
python train_ippo_lstm_trap.py --track --seed=2 --wandb-group=full &

python train_ippo_lstm_trap.py --track --seed=3 --wandb-group=no &
python train_ippo_lstm_trap.py --track --seed=3 --wandb-group=cognitive &
python train_ippo_lstm_trap.py --track --seed=3 --wandb-group=affective &
python train_ippo_lstm_trap.py --track --seed=3 --wandb-group=full &

python train_ippo_lstm_trap.py --track --seed=4 --wandb-group=no &
python train_ippo_lstm_trap.py --track --seed=4 --wandb-group=cognitive &
python train_ippo_lstm_trap.py --track --seed=4 --wandb-group=affective &
python train_ippo_lstm_trap.py --track --seed=4 --wandb-group=full &

python train_ippo_lstm_trap.py --track --seed=5 --wandb-group=no &
python train_ippo_lstm_trap.py --track --seed=5 --wandb-group=cognitive &
python train_ippo_lstm_trap.py --track --seed=5 --wandb-group=affective &
python train_ippo_lstm_trap.py --track --seed=5 --wandb-group=full &

wait

echo "ALL DONE"
