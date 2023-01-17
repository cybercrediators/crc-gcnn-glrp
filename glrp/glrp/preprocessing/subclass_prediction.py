import rpy2.robjects as robjects
from glrp.preprocessing.rbridge import RBridge


def subtype_prediction(f_path, output_fname):
    """
    Call the R preprocessing script and preprocess the data
    in the given file path.
    """
    rb = RBridge('./glrp/preprocessing/subtype_prediction.R', 'predict_subtypes')
    rb.call(f_path, output_fname)
