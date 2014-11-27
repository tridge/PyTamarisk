#!/usr/bin/python

import numpy, os, time, threading, Queue, cv, sys

from PyTamarisk.tamarisk import tamarisk

from optparse import OptionParser
parser = OptionParser("tam_capture.py [options]")
(opts, args) = parser.parse_args()

h = tamarisk.open()

im = numpy.zeros((480,640),dtype='uint16')
counter = 0
while True:
  tamarisk.capture(0, 1000, im)
  fname = 'raw%u.pgm' % counter
  counter += 1
  tamarisk.save_pgm(fname, im)
  print("Captured %s" % fname)
  
tamarisk.close(h)
