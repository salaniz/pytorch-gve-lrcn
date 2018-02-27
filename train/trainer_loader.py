from .lrcn_trainer import LRCNTrainer
from .gve_trainer import GVETrainer
from .sentence_classifier_trainer import SCTrainer

class TrainerLoader:
    lrcn = LRCNTrainer
    gve = GVETrainer
    sc = SCTrainer
