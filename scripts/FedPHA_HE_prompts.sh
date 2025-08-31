# bash scripts/FedPHA_HE_prompts.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

DATA="/data/fcy_data"
TRAINER=GL_SVDMSE_HE
DATASET="Office31"
NCTX=16
USERS=6
SEED=2
PROMPT_RANGES=(4 8 12 16 20 24 28 32)
GPU_LOAD=(1 1 1 1 1 1 1 1)
NUM_GPUS=8

check_path() {
  OUTPUT_DIR="output/${DATASET}/${TRAINER}/specify_True/beta_0.5/ep1_r50/alpha1.0_ratio0.8/prompts_${1}/seed_${SEED}/para1"
  if [ -f "${OUTPUT_DIR}/acc.csv" ]; then
    echo "Skipping ${OUTPUT_DIR}, acc.csv already exists."
    return 1
  else
    return 0
  fi
}

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
