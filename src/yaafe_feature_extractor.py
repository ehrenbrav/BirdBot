from yaafelib import *
import numpy as np

def extract(sample_frequency, data):

    # Create the Feature Plan to extract the features
    # we want.
    fp = FeaturePlan(sample_frequency)
    fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
    
    # Create the Data Flow.
    df = fp.getDataFlow()

    # Configure the YAAFE Engine.
    engine = Engine()
    engine.load(df)

    # Ensure data is float64.
    data = data.astype('float64')
    
    # Ensure it is of shape (1, len(data)).
    data = np.reshape(data, (1, len(data)))

    # Extract features.
    return engine.processAudio(data)
