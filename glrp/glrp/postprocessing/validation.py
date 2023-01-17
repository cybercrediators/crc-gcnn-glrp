from glrp.preprocessing.rbridge import RBridge
from pathlib import Path

def wgcna(f_path, output_fname):
    """
    Call the R wgcna script and perform
    wgcna on the input data and save it to the given
    output path.
    """
    rb = RBridge('./wgcna.R', 'perform_wgcna')
    rb.call(f_path, output_fname)
