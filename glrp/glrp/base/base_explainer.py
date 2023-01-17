class BaseExplainer(object):
    '''Abstract base class for a gcnn explainer'''
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

    def explain(self):
        '''Explain a given model.'''
        raise NotImplementedError

    def plot_things(self):
        '''do some plots'''
        raise NotImplementedError

    def visualize(self):
        '''do visualizations'''
        raise NotImplementedError
