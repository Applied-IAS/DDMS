class LossNanError(Exception):
    def __init__(self):
        Exception.__init__(self)
        self.message='Loss nan'