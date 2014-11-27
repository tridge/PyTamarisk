#!/usr/bin/python

import numpy, os, time, threading, Queue, cv, sys

from PyTamarisk.tamarisk import tamarisk
from PyTamarisk import TAMSerial

from optparse import OptionParser
parser = OptionParser("tam_capture.py [options]")
parser.add_option("--port", default=None, help="serial control port")
parser.add_option("--baudrate", type='int',
                  help="serial baud rate", default=57600)
parser.add_option("--video-orientation", type='int', default=TAMSerial.VID_ORIENT_NORMAL)
parser.add_option("--video-source", type='int', default=TAMSerial.VID_SRC_14)
parser.add_option("--agc-mode", type='int', default=TAMSerial.AGC_MODE_AUTO)
parser.add_option("--ice-enable", type='int', default=1)
parser.add_option("--ice-strength", type='int', default=3)
parser.add_option("--white-hot", type='int', default=1)
parser.add_option("--colorization-enable", type='int', default=0)
(opts, args) = parser.parse_args()


def TAMSetup(dev):
  '''setup camera via serial'''
  dev.send_message(MSG_VER_GET)
  dev.wait_ack()
  dev.send_uint16(MSG_COLORIZATION_ENABLE, opts.colorization_enable)
  dev.wait_ack()
  dev.send_uint16(MSG_VIDEO_ORIENTATION, opts.video_orientation)
  dev.wait_ack()
  dev.send_uint16(MSG_VIDEO_SOURCE, opts.video_source)
  dev.wait_ack()
  dev.send_uint16(MSG_AGC_MODE, opts.agc_mode)
  dev.wait_ack()
  dev.send_uint16(MSG_ICE_ENABLE, opts.ice_enable)
  dev.wait_ack()
  dev.send_uint16(MSG_ICE_STRENGTH, opts.ice_strength)
  dev.wait_ack()
  if opts.white_hot == 1:
    dev.send_message(MSG_WHITE_HOT)
  else:
    dev.send_message(MSG_BLACK_HOT)
  dev.wait_ack()

ser = None
if opts.port:
  ser = TAMSerial.TAMSerial(opts.port, baudrate=opts.baudrate, timeout=2)
  TAMSetup(ser)
  
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
