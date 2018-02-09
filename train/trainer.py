from .lrcn_trainer import LRCNTrainer

class Trainer:
    def train_epoch(self):
        raise NotImplementedError
    def train_step(self):
        raise NotImplementedError
    def eval_step(self):
        raise NotImplementedError

