#!/usr/bin/python

import numpy, os, time, threading, Queue, cv, sys

from PyTamarisk.tamarisk import tamarisk
from PyTamarisk import TAMSerial

from optparse import OptionParser
parser = OptionParser("tam_capture.py [options]")
parser.add_option("--port", default=None, help="serial control port")
parser.add_option("--baudrate", type='int',
                  help="serial baud rate", default=57600)
parser.add_option("--rate", type='float', default=2.0)
parser.add_option("--video-orientation", type='int', default=TAMSerial.VID_ORIENT_NORMAL)
parser.add_option("--video-source", type='int', default=TAMSerial.VID_SRC_14)
parser.add_option("--agc-mode", type='int', default=TAMSerial.AGC_MODE_AUTO)
parser.add_option("--ice-enable", type='int', default=1)
parser.add_option("--ice-strength", type='int', default=3)
parser.add_option("--white-hot", type='int', default=1)
parser.add_option("--colorization-enable", type='int', default=0)
parser.add_option("--tcomp-disable", type='int', default=0)
(opts, args) = parser.parse_args()


def TAMSetup(dev):
  '''setup camera via serial'''
  dev.send_message(TAMSerial.MSG_VER_GET)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_COLORIZATION_ENABLE, opts.colorization_enable)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_VIDEO_ORIENTATION, opts.video_orientation)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_VIDEO_SOURCE, opts.video_source)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_AGC_MODE, opts.agc_mode)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_ICE_ENABLE, opts.ice_enable)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_ICE_STRENGTH, opts.ice_strength)
  dev.wait_ack()
  dev.send_uint16(TAMSerial.MSG_TCOMP_DISABLE, opts.tcomp_disable)
  dev.wait_ack()
  if opts.white_hot == 1:
    dev.send_message(TAMSerial.MSG_WHITE_HOT)
  else:
    dev.send_message(TAMSerial.MSG_BLACK_HOT)
  dev.wait_ack()

def frame_time(t):
    '''return a time string for a filename with 0.01 sec resolution'''
    # round to the nearest 100th of a second
    t += 0.005
    hundredths = int(t * 100.0) % 100
    return "%s%02uZ" % (time.strftime("%Y%m%d%H%M%S", time.gmtime(t)), hundredths)

ser = None
if opts.port:
  ser = TAMSerial.TAMSerial(opts.port, baudrate=opts.baudrate, timeout=2)
  TAMSetup(ser)
  tcomp_log = open("tcomp.log", "w")
  tcomp_log.write("Filename\tNRanges\tFlags\tRange\tMin\tMax\tTComp\n")
h = tamarisk.open()

im = numpy.zeros((480,640),dtype='uint16')
tlast = 0
counter = 0
tcomp = None
while True:
  tamarisk.capture(0, 1000, im)
  if time.time() - tlast < 1.0/opts.rate:
    continue
  tlast = time.time()
  fname = 'raw%s.pgm' % frame_time(time.time())
  counter += 1
  tamarisk.save_pgm(fname, im)
  if ser is not None:
    tcomp_new = ser.get_tcomp()
    if tcomp_new is not None:
      tcomp = tcomp_new
    if tcomp is not None:
      print(tcomp)
      tcomp_log.write("%s\t%u\t%u\t%u\t%u\t%u\t%u\n" % (fname,
                                                        tcomp.nranges,
                                                        tcomp.flags,
                                                        tcomp.range,
                                                        tcomp.min,
                                                        tcomp.max,
                                                        tcomp.tcomp))
  print("Captured %s - %s" % (fname, str(tcomp)))
  
  
tamarisk.close(h)
