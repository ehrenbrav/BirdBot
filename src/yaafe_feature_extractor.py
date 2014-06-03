from yaafelib import *
import numpy as np

def extract(sample_frequency, data):

    # Set some constants.
    FRAME_SIZE = 1024
    STEP_SIZE = 512

    # Create the Feature Plan to extract the features
    # we want.
    fp = FeaturePlan(sample_frequency)
    fp.addFeature('mfcc: MFCC blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature('autocorrelation: AutoCorrelation blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature('flatness: SpectralFlatness FFTWindow=Hamming blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature('flux: SpectralFlux FFTWindow=Hamming blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature('rolloff: SpectralRolloff FFTWindow=Hamming blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature('variation: SpectralVariation FFTWindow=Hamming blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature('zcr: ZCR blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))

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
