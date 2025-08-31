#!/bin/bash
# =============================================================================
# Script: FedPHA_HE_prompts.sh
# Purpose:
#   Sweep heterogeneous prompt-length combinations for Office31 with trainer
#   GL_SVDMSE_HE. Each combo uses the pattern [A, A, B, B, C, C] across 6 users,
#   where A/B/C are drawn from PROMPT_RANGES.
#
# Key features:
#   - Multi-GPU dispatch with per-GPU concurrency caps (GPU_LOAD).
#   - Auto-skip finished runs via presence of acc.csv (idempotent / resume-able).
#   - Human-readable logs for progress tracking.
#
# Usage:
#   bash scripts/FedPHA_HE_prompts.sh
# =============================================================================

# -----------------------------
# Device & Debug
# -----------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1   # helpful for debugging CUDA stack traces

# -----------------------------
# Experiment Config
# -----------------------------
DATA="/data/fcy_data"          # (reserved) data root; set/used inside python if needed
TRAINER=GL_SVDMSE_HE           # trainer name
DATASET="Office31"             # dataset name
NCTX=16                        # base prompt context length (method may override by --prompts_lens)
USERS=6                        # number of federated users
SEED=2                         # random seed

# Candidate prompt lengths; will form A/B/C choices for [A,A,B,B,C,C]
PROMPT_RANGES=(4 8 12 16 20 24 28 32)

# Per-GPU concurrency caps. Length must match NUM_GPUS.
GPU_LOAD=(1 1 1 1 1 1 1 1)
NUM_GPUS=8

# -----------------------------
# Helper: check if result exists
# Skips a combo when acc.csv is present (supports resume)
# -----------------------------
check_path() {
  OUTPUT_DIR="output/${DATASET}/${TRAINER}/specify_True/beta_0.5/ep1_r50/alpha1.0_ratio0.8/prompts_${1}/seed_${SEED}/para1"
  if [ -f "${OUTPUT_DIR}/acc.csv" ]; then
    echo "Skipping ${OUTPUT_DIR}, acc.csv already exists."
    return 1
  else
    return 0
  fi
}

# -----------------------------
# GPU scheduling state
# gpu_index: current GPU id (0..NUM_GPUS-1)
# current_gpu_load: number of active tasks on the current GPU
# -----------------------------
gpu_index=0
current_gpu_load=0

for DOMAIN1_PROMPT in ${PROMPT_RANGES[@]}; do
  for DOMAIN2_PROMPT in ${PROMPT_RANGES[@]}; do
    for DOMAIN3_PROMPT in ${PROMPT_RANGES[@]}; do
      PROMPTS_LENS=("${DOMAIN1_PROMPT}" "${DOMAIN1_PROMPT}" "${DOMAIN2_PROMPT}" "${DOMAIN2_PROMPT}" "${DOMAIN3_PROMPT}" "${DOMAIN3_PROMPT}")
      PROMPT_LENS_STR=$(IFS=_; echo "${PROMPTS_LENS[*]}")

      check_path "${PROMPT_LENS_STR}"
      if [ $? -eq 1 ]; then
        continue
      fi

      while [ "${GPU_LOAD[$gpu_index]}" -eq 0 ]; do
        gpu_index=$((gpu_index + 1))
        current_gpu_load=0

        if [ "$gpu_index" -ge "$NUM_GPUS" ]; then
          echo "All GPUs are fully utilized or not allowed to run tasks. Exiting script."
          exit 0
        fi
      done

      if [ "${current_gpu_load}" -ge "${GPU_LOAD[$gpu_index]}" ]; then
        gpu_index=$((gpu_index + 1))
        current_gpu_load=0

        if [ "$gpu_index" -ge "$NUM_GPUS" ]; then
          echo "All GPUs are fully utilized. Exiting script."
          exit 0
        fi
      fi

      echo "Running experiment: Prompts = ${PROMPT_LENS_STR} on GPU ${gpu_index} (Task ${current_gpu_load}/${GPU_LOAD[$gpu_index]})"

      python federated_main.py \
        --trainer ${TRAINER} \
        --dataset ${DATASET} \
        --device_id ${gpu_index} \
        --n_ctx ${NCTX} \
        --num_users ${USERS} \
        --seed ${SEED} \
        --specify True \
        --prompts_lens "${PROMPTS_LENS[@]}" &

      current_gpu_load=$((current_gpu_load + 1))
      sleep 1
    done
  done
done

wait
echo "All experiments completed."
