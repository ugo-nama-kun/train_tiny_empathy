
for i in `seq 1 10`
do
  python train_grid_rooms_inference_analysis.py --track --enable-empathy &
  python train_grid_rooms_inference_analysis.py --track &
  wait
done
echo "ALL DONE"
