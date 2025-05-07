
for i in `seq 0 19`
do
  # full empathy
  #python train_foodshare_decoder_learning.py --track --decoding-mode="full" --enable-learning --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect"  --enable-learning --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="full" --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect" --enable-empathy &

  # affective empathy
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect" --enable-learning &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect" &

  # cognitive empathy
  #python train_foodshare_decoder_learning.py --track --decoding-mode="full" --weight-empathy=0.0 --enable-learning --enable-empathy &
  #python train_foodshare_decoder_learning.py --track --decoding-mode="affect" --weight-empathy=0.0 --enable-learning --enable-empathy &
  python train_foodshare_decoder_learning.py --track --decoding-mode="full" --weight-empathy=0.0 --enable-empathy &
  python train_foodshare_decoder_learning.py --track --decoding-mode="affect" --weight-empathy=0.0 --enable-empathy &
  wait
done
echo "ALL DONE"
