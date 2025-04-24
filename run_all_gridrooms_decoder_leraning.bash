
for i in `seq 0 19`
do
  python train_grid_rooms_decoder_learning.py --cuda --track --decoding-mode="full" --enable_learning &
  python train_grid_rooms_decoder_learning.py --cuda --track --decoding-mode="affect"  --enable_learning &
  python train_grid_rooms_decoder_learning.py --track --decoding-mode="full" &
  python train_grid_rooms_decoder_learning.py --track --decoding-mode="affect" &
  wait
done
echo "ALL DONE"
