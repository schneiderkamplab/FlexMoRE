#!/bin/bash
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_COMPILE_WORKERS=1
export TORCHINDUCTOR_COMPILE_WORKER_KIND=spawn
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
export TRITON_CACHE_DIR=/tmp/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(torchrun \
  --nproc-per-node=4 \
  src/scripts/train/OLMoE-2x7B-anneal.py \
  flex_danish_v1 \
  --trainer.callbacks.profiler.enabled=true \
  --dataset.mix_base_dir=data \
  --dataset.mix=danish \
  --trainer.max_duration.value=50_000_000_000 \
  --trainer.max_duration.unit=tokens \
  --trainer.load_path=public/Flex-public-2x7B-1T-v1 \
  --model.block.feed_forward_moe.router.top_k=2 \
  --train_module.rank_microbatch_size=8192 \
  --train_module.scheduler.warmup_steps=2000 \
  --train_module.optim.lr=9e-4 \
  --trainer.save_folder=checkpoints/Flex-danish-2x7B-1T-v1 \
  --dataset.work_dir=/tmp/data \
  --train_module.compile_model=true \
  --data_loader.global_batch_size=524288 \
  --train_module.max_sequence_length=4096 \
  --dataset.sequence_length=4096 \
2>&1) | tee -a flex_danish_v1.log
