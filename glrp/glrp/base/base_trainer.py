class BaseTrainer:
    '''Abstract class for the model runner'''

    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

    def summarize(self):
        '''log current values to e.g. tensorboard'''
        raise NotImplementedError

    def run(self):
        '''run the given model'''
        raise NotImplementedError
