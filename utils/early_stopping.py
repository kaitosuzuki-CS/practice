import copy


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.best_loss = float("inf")
        self.best_model = None
        self.stop = False

    def __call__(self, model, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
