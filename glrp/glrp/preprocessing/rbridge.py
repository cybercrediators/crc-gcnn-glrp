import rpy2.robjects as robjects


class RBridge(object):

    """
    Call and use functions from existing
    R scripts
    """

    def __init__(self, script_path, function_name):
        """Init r script and retrieve the given function"""
        self.script_path = script_path
        self.function_name = function_name

        try:
            r = robjects.r
            r['source'](script_path)
            self.function = robjects.globalenv[function_name]
        except Exception as e:
            print("Error creating the r object or function not found!")
            raise e

    def call(self, *args):
        return self.function(*args)
