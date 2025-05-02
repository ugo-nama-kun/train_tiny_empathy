
for i in `seq 0 19`
do
  # w/ cognitive empathy
  #python train_foodshare_decoder_learning.py --track --decoding-mode="full" --enable-learning --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect"  --enable-learning --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="full" --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect" --enable-empathy &

  # w/o cognitive empathy
  python train_foodshare_decoder_learning.py --track --decoding-mode="affect" --enable-learning &
  python train_foodshare_decoder_learning.py --track --decoding-mode="affect" &
  wait
done
echo "ALL DONE"
