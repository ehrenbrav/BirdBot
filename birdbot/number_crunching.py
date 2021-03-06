"""
Functions for doing the actual training/testing.
"""
import birdbot.params as p
import birdbot.fileIO
import numpy as np
import logging
import time

# pylint: disable=C0103

def run_calculation(data, functions, bk, layers):
    """Do the actual work."""

    # Loop through the chunks of data.
    for x, y in data.train_set_list:

        # Load this bit of data into our shared variables, if not already there.
        if len(data.train_set_list) > 1:
            data.shared_train_x.set_value(x)
            data.shared_train_y.set_value(y)

        # Calculate the number of minibatches to run through.
        n_batches = data.shared_train_x.get_value().shape[0] / p.BATCH_SIZE

        # Loop through the minibatches.
        for minibatch_index in xrange(n_batches):

            # Train the model.
            functions.train_model(minibatch_index)

            # Increment the iteration number.
            bk.iteration += 1

            # Adjust validation frequency.
            validation_frequency = min(n_batches, bk.patience / 2)

            # If the time is right, calculate validation and/or test scores.
            if (bk.iteration + 1) % (validation_frequency * 2) == 0:

                # Run the accuracy calculation.
                __compute_accuracy__(
                    functions, data, bk, minibatch_index, n_batches, layers)

def __compute_accuracy__(
        functions, data, bk, minibatch_index, n_train_batches, layers):
    """
    Run the model on the validation
    and testing sets (if applicable).
    """

    # Calculate the validation score.
    validation_losses = __test_model__(
        data.valid_set_list,
        data.shared_valid_x,
        data.shared_valid_y,
        functions.validate_model)
    this_validation_loss = np.mean(validation_losses)

    message = 'Epoch %i, minibatch %i/%i, validation error %f %%' % \
          (bk.epoch, minibatch_index + 1, n_train_batches,
           this_validation_loss * 100)
    logging.info(message)

    # Calculate the training model score once in a while.
    if (bk.epoch % 5 == 0) and p.PRINT_TRAINING_SET_ERROR:
        training_losses = __test_model__(
            data.train_set_list,
            data.shared_train_x,
            data.shared_train_y,
            functions.test_training_model)
        this_training_loss = np.mean(training_losses)

        message = 'Training set error: %f %%' % \
          (this_training_loss * 100)
        logging.info(message)

    # If we have the best validation score so far...
    if this_validation_loss < bk.best_validation_loss:

        # Improve patience if the gain is big enough.
        if this_validation_loss < bk.best_validation_loss * \
          p.IMPROVEMENT_THRESHOLD:

            bk.patience = max(
                bk.patience, bk.iteration * p.PATIENCE_INCREASE)

        # Record new high score.
        bk.best_validation_loss = this_validation_loss

        # Save our params.
        init_params = []
        for layer in layers:
            init_params.append(
                [layer.params[0].get_value(), layer.params[1].get_value()])
        birdbot.fileIO.save_model(bk, init_params)

        # Note the time.
        elapsed_time = time.clock() - bk.start_time + bk.total_time
        bk.total_time = elapsed_time
        bk.start_time = time.clock()
        time_string = "Elapsed Time: %.1fs" % elapsed_time
        logging.info(time_string)

        # Try the test set.
        test_losses = __test_model__(
            data.test_set_list,
            data.shared_test_x,
            data.shared_test_y,
            functions.test_model)
        bk.test_score = np.mean(test_losses)

        message = '  Epoch %i, minibatch %i/%i, best test error: %f %%' % \
          (bk.epoch,
           minibatch_index + 1,
           n_train_batches,
           bk.test_score * 100.)
        logging.info(message)

def __test_model__(data_list, shared_x, shared_y, function):
    """Run the model on the validation or testing set."""

    # Loop through all the data chunks in our set.
    losses = []
    for x, y in data_list:

        # Update the shared variables.
        if len(data_list) > 1:
            shared_x.set_value(x)
            shared_y.set_value(y)

        # Calculate the number of minibatches we need.
        n_batches = shared_x.get_value().shape[0] / p.BATCH_SIZE

        # Loop through the minibatches.
        for minibatch in xrange(n_batches):

            # Run the calculation.
            losses.append(function(minibatch))

    return losses

