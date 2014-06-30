"""
This module inputs a single frame
and yields a class along with the
confidence the classifier has in its
classification.
"""

import weka.core.serialization as serialization
import weka.core.jvm as jvm
from weka.classifiers import Classifier as weka_classifier
from weka.core.dataset import Instance
import classifier_writer as cw

class Classifier:
    """
    This class is a Weka classifier, read
    from memory. Obviously, the
    data handed to this classifier must
    have the same format as training
    data used to create the classifier.
    """

    classifier = None

    def __init__(self):
        """
        Start the JVM and
        read the classifier into
        memory.
        """

        # Start the JVM.
        jvm.start()

        # Read the classifier into memory.
        objects = serialization.read_all(cw.CLASSIFIER_NAME)
        self.classifier = weka_classifier(jobject=objects[0])

    def classify_frame(self, frame_features):

        # Convert to an instance.
        instance = self.convert_to_instance(frame_features)

        # Classify.
        return self.classifier.classify_instance(instance)

def convert_to_instance(self, frame_features):

    # TODO read dataset into the instance.
    return Instance.create_instance(frame_features)
