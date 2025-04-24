
for i in `seq 0 19`
do
  python train_sharefood_decoder_learning.py --track --decoding-mode="full" --enable_learning &
  python train_sharefood_decoder_learning.py --track --decoding-mode="affect"  --enable_learning &
  python train_sharefood_decoder_learning.py --track --decoding-mode="full" &
  python train_sharefood_decoder_learning.py --track --decoding-mode="affect" &
  wait
done
echo "ALL DONE"
