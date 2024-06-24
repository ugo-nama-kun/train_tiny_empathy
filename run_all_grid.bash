
for i in `seq 0 19`
do
  python train_grid_rooms.py --track --enable-empathy --weight-empathy 0.5 &
  python train_grid_rooms.py --track --enable-empathy &
  python train_grid_rooms.py --track --weight-empathy 0.5 &
  python train_grid_rooms.py --track
done
echo "ALL DONE"