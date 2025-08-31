# bash scripts/FedPHA_few_shot.sh
# trainers = ['PROMPTFL', 'PROMPTFL_PROX', 'FEDPGP', 'GLP_OT', 'GL_SVDMSE']
# python federated_main.py --trainer GL_SVDMSE --dataset caltech101

TRAINER="GL_SVDMSE"     # choose from the list above
DATASET="caltech101"    # dataset name
SHOTS=2                 # number of shots per class
BACKBONE="rn50"         # backbone model: rn50 | vit_b16 | ...
USERS=10                # number of federated clients
SEED=1                  # random seed

python federated_main.py \
  --trainer ${TRAINER} \
  --dataset ${DATASET} \
  --shots ${SHOTS} \
  --backbone ${BACKBONE} \
  --num_users ${USERS} \
  --seed ${SEED} \
