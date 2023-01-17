class BaseModel(object):
    '''Abstract base class for a tf model'''

    def __init__(self, config):
        self.config = config
        # TODO: init logger

    def save(self):
        '''Save the current model'''
        # save the current model
        raise NotImplementedError

    def load(self):
        '''Load a saved model'''
        # load the current model
        raise NotImplementedError

    def build_model(self):
        '''Build a new model'''
        raise NotImplementedError

    def evaluate(self):
        '''Evaluate the current model'''
        raise NotImplementedError

    def train(self):
        '''Train the model'''
        raise NotImplementedError
