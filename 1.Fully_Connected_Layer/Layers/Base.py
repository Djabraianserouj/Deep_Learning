class BaseLayer:
    # weights = []

    def __init__(self):
        self.trainable = False #Trainable layers have parameters that are optimized during training
                              # the non trainable layers remained fixed