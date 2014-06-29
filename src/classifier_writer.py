#!/usr/bin/env python
"""
This module either takes an ARFF
file or a directory with training
data and outputs a WEKA classifier.
"""

import os
import arff_writer
import weka.core.serialization as serialization
from weka.core.dataset import Instances
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
import weka.core.jvm as jvm

# Name of classifier.
CLASSIFIER_NAME = 'classifier.weka'

def write_classifier(path):
    """
    Use either an ARFF file or the
    directory to the training data
    to write the classifier.
    """

    # Handle directory case.
    if os.path.isdir(path):
        arff_writer.write_arff(path)
        process_arff(arff_writer.ARFF_NAME)
    else:
        process_arff(path)

def process_arff(path):
    """
    This function converts the ARFF file to a
    weka classifier.
    """

    # Load the ARFF file.
    loader = Loader("weka.core.converters.ArffLoader")
    training_data = loader.load_file(path)

    # Tell weka that the class is indicated last.
    training_data.set_class_index(training_data.num_attributes() - 1)

    # Build the classifier.
    classifier = Classifier(classname="weka.classifiers.trees.RandomForest")
    classifier.build_classifier(training_data)

    # Save the classifier.
    serialization.write_all(CLASSIFIER_NAME,
                            [classifier,
                             Instances.template_instances(training_data)])

    # Cross-validate.
    evaluation = Evaluation(training_data)
    evaluation.crossvalidate_model(classifier, training_data, 10, Random(42))
    print evaluation.to_summary()
    print evaluation.to_class_details()
    print evaluation.to_matrix()

if __name__ == '__main__':

    import argparse

    # Get the name of the directory/ARFF file
    # specified on the command line.
    # pylint: disable=C0103
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    # Start the JVM stuff.
    try:
        jvm.start()
        write_classifier(parser.parse_args().path)
    except Exception, e:
        print e
    finally:
        jvm.stop()

