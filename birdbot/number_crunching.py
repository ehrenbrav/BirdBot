"""
Functions for doing the actual training/testing.
"""
import birdbot.paramp
# pylint: disable=C0103

def run_calculation(data, functions, bk):
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
            bk.validation_frequency = min(n_batches, bk.patience / 2)

            # If the time is right, calculate validation and/or test scores.
            if (bk.iteration + 1) % bk.validation_frequency == 0:

                # Run the accuracy calculation.
                compute_accuracy(
                    functions, data, bk, minibatch_index, n_batches)

def __compute_accuracy__(functions, data, bk, minibatch_index, n_train_batches):
    """
    Run the model on the validation
    and testing sets (if applicable).
    """

    # Calculate the validation score.
    validation_losses = test_model(
        data.valid_set_list,
        data.shared_valid_x,
        data.shared_valid_y,
        functions.validate_model)
    this_validation_loss = np.mean(validation_losses)

    print('Epoch %i, minibatch %i/%i, validation error %f %%' %\
          (bk.epoch, minibatch_index + 1, n_train_batches,
           this_validation_loss * 100))

    # If we have the best validation score so far...
    if this_validation_loss < bk.best_validation_loss:

        # Improve patience if the gain is big enough.
        if this_validation_loss < bk.best_validation_loss * \
          p.IMPROVEMENT_THRESHOLD:

            bk.patience = max(
                bk.patience, bk.iteration * p.PATIENCE_INCREASE)

            bk.best_validation_loss = this_validation_loss

            # Try the test set.
            test_losses = test_model(
                data.test_set_list,
                data.shared_test_x,
                data.shared_test_y,
                functions.test_model)
            bk.test_score = np.mean(test_losses)

            print(('  Epoch %i, minibatch %i/%i, '
                  'test error of best '
                  'model %f %%')
                  % (bk.epoch,
                  minibatch_index + 1,
                  n_train_batches,
                  bk.test_score * 100.))

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

