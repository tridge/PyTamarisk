#!/usr/bin/python

import numpy, os, time, threading, Queue, cv, sys

from PyTamarisk.tamarisk import tamarisk

from optparse import OptionParser
parser = OptionParser("tam_capture.py [options]")
(opts, args) = parser.parse_args()

h = tamarisk.open()
tamarisk.close(h)
