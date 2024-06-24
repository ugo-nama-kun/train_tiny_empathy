
for i in `seq 0 3`
do
  python train_sharefood.py --track --enable-empathy --weight-empathy 0.5 &
  python train_sharefood.py --track --enable-empathy &
  python train_sharefood.py --track --weight-empathy 0.5 &
  python train_sharefood.py --track
done
echo "ALL DONE"
