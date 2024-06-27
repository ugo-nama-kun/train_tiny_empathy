for i in `seq 0 19`
do
  #python train_trap.py --track --cuda --enable-empathy --weight-empathy 0.5
  #python train_trap.py --track --cuda --enable-empathy 
  #python train_trap.py --track --cuda --weight-empathy 0.5
  python train_trap.py --track --cuda
done
echo "ALL DONE"
