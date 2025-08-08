from Dassl.dassl.utils import Registry, check_availability
from trainers.CLIP import CLIP
from trainers.GLP_OT import GLP_OT
from trainers.FEDPGP import FEDPGP
from trainers.PROMPTFL import PROMPTFL
from trainers.GL_SVDMSE import GL_SVDMSE
from trainers.GL_SVDMSE_HE import GL_SVDMSE_HE

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(GLP_OT)
TRAINER_REGISTRY.register(FEDPGP)
TRAINER_REGISTRY.register(PROMPTFL)
TRAINER_REGISTRY.register(GL_SVDMSE)
TRAINER_REGISTRY.register(GL_SVDMSE_HE)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
