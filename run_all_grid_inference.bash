
for i in `seq 1 10`
do
  python train_grid_rooms_inference_analysis.py --track --enable-empathy --enable-inference &
  python train_grid_rooms_inference_analysis.py --track --enable-inference &
  wait
done
echo "ALL DONE"
