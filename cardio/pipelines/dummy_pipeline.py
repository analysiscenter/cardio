import sys
import os
sys.path.append(os.path.join(".."))

import cardio.dataset as ds

def dummy_pipeline():
    return (ds.Pipeline()
            .load(fmt="wfdb", components=["signal", "meta"])
            .random_resample_signals("normal", loc=300, scale=10)
            .drop_short_signals(4000)
            .split_signals(3000, 3000))