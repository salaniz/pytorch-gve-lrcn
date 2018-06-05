import torch

from .lrcn import LRCN
from .gve import GVE
from .sentence_classifier import SentenceClassifier

class ModelLoader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def lrcn(self):
        # LRCN arguments
        pretrained_model = self.args.pretrained_model
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)

        layers_to_truncate = self.args.layers_to_truncate

        lrcn = LRCN(pretrained_model, embedding_size, hidden_size, vocab_size,
                layers_to_truncate)

        return lrcn

    def gve(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # GVE arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        input_size = self.dataset.input_size
        num_classes = self.dataset.num_classes

        sc = self.sc()
        sc.load_state_dict(torch.load(self.args.sc_ckpt))
        for param in sc.parameters():
            param.requires_grad = False
        sc.eval()

        gve = GVE(input_size, embedding_size, hidden_size, vocab_size, sc,
                num_classes)

        if self.args.weights_ckpt:
            gve.load_state_dict(torch.load(self.args.weights_ckpt))

        return gve



    def sc(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # Sentence classifier arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        num_classes = self.dataset.num_classes

        sc = SentenceClassifier(embedding_size, hidden_size, vocab_size,
                num_classes)

        return sc
