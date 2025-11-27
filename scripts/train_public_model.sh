# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
export MODEL_SIZE="tiny" # tiny or 7B
export DATA_ROOT="http://flexolmo-data.org"
export CHECKPOINTS=checkpoints  # /path/to/checkpoints

# Train base model on public mix for 1T tokens (schedule set to 5T)
torchrun --nproc-per-node=8 ./src/scripts/train/OLMo2-${MODEL_SIZE}-train.py OLMo2-${MODEL_SIZE}-public-mix \
    --dataset.mix_base_dir=${DATA_ROOT} \
    --dataset.mix=public_mix \
    --train_module.float8_config.enabled=true \
    --trainer.duration.value=5_000_000_000_000 \
    --trainer.duration.unit=tokens \
    --trainer.hard_stop.value=1_000_000_000_000 \
    --trainer.hard_stop.unit=tokens \
    --trainer.save_folder=${CHECKPOINTS}/OLMo2-${MODEL_SIZE}-public-mix

# Anneal for 50B tokens, using the checkpoint from the base model training.
# Make sure to set the correct path to the checkpoint.
BASE_CHECKPOINT=${CHECKPOINTS}/OLMo2-${MODEL_SIZE}-public-mix/step238419
torchrun --nproc-per-node=8 ./src/scripts/train/OLMo2-anneal.py OLMo2-${MODEL_SIZE}-anneal-public-mix ${BASE_CHECKPOINT}
    --dataset.mix_base_dir=${DATA_ROOT} \
    --dataset.mix=public_mix \
    --train_module.float8_config.enabled=true \
    --trainer.duration.value=50_000_000_000\
    --trainer.duration.unit=tokens \
    --trainer.save_folder=${CHECKPOINTS}/OLMo2-7B-anneal-public-mix

# Unshard the final checkpoint
MODEL_PATH=${CHECKPOINTS}/OLMo2-${MODEL_SIZE}-anneal-public-mix/step11921
python src/scripts/utils/unshard.py -i ${MODEL_PATH} -o ${MODEL_PATH}-unsharded

# Upcycle the dense model to 2xMODEL_SIZE (when not using domain-specific embeddings)
MODEL_PATH=${CHECKPOINTS}/OLMo2-${MODEL_SIZE}-anneal-public-mix/step11921
python src/scripts/upcycle/dense_to_expert_moe.py -m ${MODEL_PATH}-unsharded ${MODEL_PATH}-unsharded -t ${CHECKPOINTS}/olmoe-2x${MODEL_SIZE}-public-public