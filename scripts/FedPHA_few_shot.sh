# bash scripts/FedPHA_few_shot.sh
# trainers = ['PROMPTFL', 'PROMPTFL_PROX', 'FEDPGP', 'GLP_OT', 'GL_SVDMSE']
python federated_main.py --trainer GL_SVDMSE --dataset caltech101
