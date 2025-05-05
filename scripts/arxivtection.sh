

for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data arxivtection \
    --model mistral \
    --target_num 700 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done
