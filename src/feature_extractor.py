"""
This module inputs the sample frequency
and a numpy array of raw amplitude data
and outputs a dictionary of a number of
extracted audio features.
"""

from yaafelib import FeaturePlan, Engine
import numpy as np

# Set some constants.
FRAME_SIZE = 1024
STEP_SIZE = 512
NUMBER_MFCCS = 13
NUMBER_SPECTRAL_STATS = 4

def extract(sample_frequency, data):
    """
    This is the primary function of
    this module.
    """

    # Ensure data is float64.
    data = data.astype('float64')

    # Ensure it is of shape (1, len(data)).
    data = np.reshape(data, (1, len(data)))

    # Configure the YAAFE engine.
    engine = configure_engine(sample_frequency)

    # Extract features.
    return engine.processAudio(data)

def configure_engine(sample_frequency):
    """Set up the engine to extract the features we want."""

    # pylint: disable=C0301
    # pylint: disable=C0103

    fp = FeaturePlan(sample_frequency)
    fp.addFeature(
        'mfcc: MFCC blockSize={0} stepSize={1} CepsNbCoeffs={2}'
        .format(FRAME_SIZE, STEP_SIZE, NUMBER_MFCCS))
    # fp.addFeature(
    #    'autocorrelation: AutoCorrelation blockSize={0} stepSize={1}'
    #     .format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'flatness: SpectralFlatness FFTWindow=Hamming blockSize={0} stepSize={1}'
        .format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'flux: SpectralFlux FFTWindow=Hamming blockSize={0} stepSize={1}'
        .format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'rolloff: SpectralRolloff FFTWindow=Hamming blockSize={0} stepSize={1}'
        .format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'variation: SpectralVariation FFTWindow=Hamming blockSize={0} stepSize={1}'
        .format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'zcr: ZCR blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'energy: Energy blockSize={0} stepSize={1}'.format(FRAME_SIZE, STEP_SIZE))
    fp.addFeature(
        'spectral_stats: SpectralShapeStatistics FFTWindow=Hamming blockSize={0} stepSize={1}'
        .format(FRAME_SIZE, STEP_SIZE))

    # Create the Data Flow.
    df = fp.getDataFlow()

    # Configure the YAAFE Engine.
    engine = Engine()
    engine.load(df)
    return engine
