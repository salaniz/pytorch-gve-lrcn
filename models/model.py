from .lrcn import LRCN

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
        is_factored = self.args.factored

        lrcn = LRCN(pretrained_model, embedding_size, hidden_size, vocab_size,
                layers_to_truncate, is_factored)

        return lrcn
