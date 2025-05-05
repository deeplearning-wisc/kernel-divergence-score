

# Wikipedia
for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data wikipedia \
    --model mistral \
    --target_num 500 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done

# PhilPapers
for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data philpapers \
    --model mistral \
    --target_num 500 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done

# Enron
for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data enron \
    --model mistral \
    --target_num 500 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done

# HackerNews
for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data hackernews \
    --model mistral \
    --target_num 500 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done

# PILE-CC
for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data cc \
    --model mistral \
    --target_num 500 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done

# StackExchange
for c in $(seq 0.00 0.05 1.00); do
  CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --data stackexchange \
    --model mistral \
    --target_num 500 \
    --out_dir out/ \
    --contamination $c \
    --sgd \
    --lr 0.0001 \
    --seed 0
done