# train_tiny_empathy

The training code for the tiny-empathy environments

### how to train agents
```commandline
# * can be one of {sharefood, grid_rooms, trap}
python train_*.py --enable-empathy --weight-empathy 0.5
```

Check individual training codes (train_*.py) for complete options.
`run_all_*.bash` includes typical settings.

### how to run trained agents
```commandline
# * can be one of {sharefood, grid, trap}
python run_*.py
```

Modify `model_name` and `enable_empathy` based on the weight parameters files (saved in `models/{dates}/{weight_names}.pt`) and training conditions.
if `--enable-empathy` option was enabled in the training, `enable_empathy = True`. 

#### Typical training time in my macbook Air M1

- FoodShare-v0: few minutes
- GridRooms-v0: ~30 minutes
- Trap-v0: around 6 hours or more
