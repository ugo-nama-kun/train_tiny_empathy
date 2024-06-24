for i in `seq 0 19`
do
  python train_trap.py --track --enable-empathy --weight-empathy 0.5 &
  python train_trap.py --track --enable-empathy &
  python train_trap.py --track --weight-empathy 0.5 &
  python train_trap.py --track
done
echo "ALL DONE"