class BaseLogger:
    def __init__(self, full_config):
        raise NotImplementedError("Logger is not implemented")

    def log_metrics(self, y_actual, y_pred, mode):
        raise NotImplementedError("Must be implemented in subclass")

    def log_loss(self, loss, mode):
        raise NotImplementedError("Must be implemented in subclass")

    def log_plots(self, metadata, mode):
        raise NotImplementedError("Must be implemented in subclass")

    def log_model_params(self, mode):
        raise NotImplementedError("Must be implemented in subclass")
