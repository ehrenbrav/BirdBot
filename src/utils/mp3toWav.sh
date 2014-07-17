#!/bin/bash

# Simple script to convert and save all mp3s in a directory to wav format.
for i in *.mp3; do lame --decode "$i" "`basename "$i" .mp3`".wav; done
