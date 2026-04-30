class CustomLoss:
    def __init__(self, loss_fn, device):
        self.loss = loss_fn

    def __call__(self, x_pred, x):
        return self.loss(x_pred, x)
