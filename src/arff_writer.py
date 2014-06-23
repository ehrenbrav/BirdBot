#!/usr/bin/env python
"""
This module takes a directory
of directories each of which contain
sound files, and uses the
current settings of the feature extractor
to write an ARFF file. The name of each directory
must be the same as the bird whose sounds
are recorded in the files within that directory.
"""

import argparse
import os
import feature_extractor

# Name of ARFF file.
ARFF_NAME = 'training_data.arff'

# Get name of directory specified on the command line.
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
path = parser.parse_args().path

# Handle error of no directory provided.
if not os.path.isdir(path):
    print """
    The path specified is not a directory.
    The training data must be in a series
    of directories, each within the directory
    named on the command line when invoking
    this utility.
    """
    exit(1)

# Get the list of sub-directories.
directory_list = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]

with open(ARFF_NAME, 'w') as arff_file:

    arff_file.write("@RELATION bird_calls\n")

    # Get the list of the features we'll be extracting.
    engine = feature_extractor.configure_engine(44100)
    feature_list = engine.getOutputs().keys()

    # Write attributes.
    for feature in feature_list:
        arff_file.write("@ATTRIBUTE " + feature + " NUMERIC\n")

    # Write classes.
    class_list = ""
    for class_name in directory_list:
        class_list += class_name + ","

    class_list = class_list[:-1]
    arff_file.write("@ATTRIBUTE class {" + class_list + "}\n")

    # Finish up.
    arff_file.write("@DATA\n")
    
    # Loop through all files and write into a single ARFF.
    # TODO write multiple MFCCs and spectral_stats


    

