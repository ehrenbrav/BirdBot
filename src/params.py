"""Parameters for the model."""

# Path to the dataset.
DATASET_PATH = 'dataset.pkl.gz'

# Set the maximum number of examples to copy to the GPU at one time.
MAX_DATA_SIZE = 533

# Learning rate.
LEARNING_RATE = .13

# Number of epochs to go through.
NUM_EPOCHS = 50

# Size of each minibatch.
BATCH_SIZE = 50

# Look at this many examples regardless.
PATIENCE = 5000

# Wait this much longer when a new best is found.
PATIENCE_INCREASE = 2

# An improvement of this much is considered significant.
IMPROVEMENT_THRESHOLD = 0.995

# Go through this many minibatches before validating.
VALIDATION_FREQUENCY = 20

