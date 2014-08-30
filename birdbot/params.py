"""Parameters for the model."""

# Path to the dataset.
DATASET_PATH = '../datasetHALF_STRIDE_128.pkl.gz'

# Path to the classification_map.
CLASSIFICATION_MAP_PATH = '../classification_map.txt'

# Set the maximum number of examples to copy to the GPU at one time.
# Ensure this is way bigger than the minibatch size.
# This should be as large as possible without crashing the GPU.
MAX_DATA_SIZE = 12000

# Learning rate.
LEARNING_RATE = .13

# Number of epochs to go through.
NUM_EPOCHS = 400

# Size of each minibatch.
BATCH_SIZE = 50

# Look at this many examples regardless.
PATIENCE = 19000

# Wait this much longer when a new best is found.
PATIENCE_INCREASE = 2

# An improvement of this much is considered significant.
IMPROVEMENT_THRESHOLD = 0.995

# Go through this many minibatches before validating.
VALIDATION_FREQUENCY = 100

# How long do we want our spectrograms?
SPECTROGRAM_DURATION = 4

# Stride distance beween starts of each spectrogram.
SPECTROGRAM_STRIDE = .5

# Frequency limits.
MAX_FREQUENCY = 13000
MIN_FREQUENCY = 100

# What percent of the dataset to designate for training?
PERCENT_TRAINING = .7

# Dimensions of width and height (always a square) of the graph.
SPECTROGRAM_SIDE_SIZE = 128

# Dimensions of color in spectrogram.
PIXEL_DIM = 1


