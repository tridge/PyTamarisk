#!/usr/bin/env python
'''
tamarisk serial protocol handling

Copyright Andrew Tridgell, August 2014
Released under GNU GPL version 3 or later
'''

import struct
from datetime import datetime
import time, os, sys

# protocol constants
PREAMBLE1 = 0x01

# message IDs
MSG_TXT     = 0x00
MSG_ACK     = 0x02
MSG_NACK    = 0x03
MSG_ERR     = 0x04
MSG_VALUE   = 0x45
MSG_VER_GET = 0x07
MSG_COLORIZATION_ENABLE = 0xCC
MSG_VIDEO_ORIENTATION   = 0xCF
MSG_VIDEO_SOURCE        = 0xD7
MSG_ICE_ENABLE          = 0x23
MSG_ICE_MINMAX          = 0x22
MSG_ICE_STRENGTH        = 0x1E
MSG_ICE_FREQ_THRESH     = 0x1F
MSG_AGC_MODE            = 0x2A
MSG_AGC_GAIN            = 0x32
MSG_AGC_GAIN_BIAS       = 0x82
MSG_AGC_LEVEL_BIAS      = 0x83
MSG_AGC_GAIN_LIMIT      = 0xD1
MSG_AGC_GAIN_FLATTEN    = 0xD2
MSG_BLACK_HOT           = 0x28
MSG_WHITE_HOT           = 0x29
MSG_TCOMP_DISABLE       = 0x18
MSG_TCOMP_STATUS        = 0x10

# video orientations
VID_ORIENT_NORMAL = 0
VID_ORIENT_VERTICAL_INVERT = 1
VID_ORIENT_HORIZONTAL_INVERT = 2
VID_ORIENT_HORVER_INVERT = 3

# video sources
VID_SRC_TEST_PATTERN = 0
VID_SRC_14 = 6
VID_SRC_14_FREEZE = 7
VID_SRC_AGC = 8
VID_SRC_SYMBOLOGY = 9

# AGC modes
AGC_MODE_FREEZE = 0
AGC_MODE_AUTO   = 1
AGC_MODE_MANUAL = 2

class TAMError(Exception):
    '''TAM error class'''
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg

class TAMAttrDict(dict):
    '''allow dictionary members as attributes'''
    def __init__(self):
        dict.__init__(self)

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.__dict__.has_key(name):
            # allow set on normal attributes
            dict.__setattr__(self, name, value)
        else:
            self.__setitem__(name, value)

class TAMDescriptor:
    '''class used to describe the layout of a TAM message'''
    def __init__(self, name, msg_format, fields=[]):
        self.name = name
        self.msg_format = msg_format
        self.fields = fields
	
    def unpack(self, msg):
	'''unpack a TAMMessage, creating the .fields and ._recs attributes in msg'''
        msg._fields = {}

        # unpack main message blocks. A comm
        buf = msg._buf[3:-1]
        count = 0
        msg._recs = []
        fields = self.fields[:]
        
        fmt = self.msg_format
        fmt = fmt.replace('#', "%u" % len(buf))
        size = struct.calcsize(fmt)
        if size != len(buf):
            raise TAMError("%s INVALID_SIZE=%u expected=%u" % (self.name, len(buf), size))
        f1 = list(struct.unpack(fmt, buf[:size]))
        i = 0
        while i < len(f1):
            field = fields.pop(0)
            msg._fields[field] = f1[i]
            i += 1

        msg._unpacked = True
        return

    def pack(self, msg, msg_id=None):
	'''pack a TAMMessage from the .fields and ._recs attributes in msg'''
        f1 = []
        if msg_id is None:
            msg_id = msg.msg_id()
        msg._buf = ''

        fields = self.fields[:]
        for f in fields:
            f1.append(msg._fields[fieldname])
        fmt = self.msg_format
        msg._buf = struct.pack(fmt, *tuple(f1))

        length = len(msg._buf)
        header = struct.pack('>BBB', PREAMBLE1, msg_id, length)
        msg._buf = header + msg._buf
        msg._buf += struct.pack('>B', *msg.checksum(data=msg._buf))

    def format(self, msg):
	'''return a formatted string for a message'''
        if not msg._unpacked:
            self.unpack(msg)
        ret = self.name + ': '
        for f in self.fields:
            if not f in msg._fields:
                continue
            v = msg._fields[f]
            if isinstance(v, str):
                ret += '%s="%s", ' % (f, v.rstrip(' \0'))
            else:
                ret += '%s=%s, ' % (f, v)
        return ret[:-2]
        

# list of supported message types.
msg_types = {
    MSG_TXT     : TAMDescriptor('MSG_TXT', '>#s', ['text']),
    MSG_ACK     : TAMDescriptor('MSG_ACK', '>H',  ['cmdId']),
    MSG_NACK    : TAMDescriptor('MSG_NACK','>H',  ['cmdId']),
    MSG_ERR     : TAMDescriptor('MSG_ERR', '>#s', ['errString']),
    MSG_VALUE   : TAMDescriptor('MSG_VALUE','>H', ['value']),
    MSG_VER_GET : TAMDescriptor('MSG_VER_GET','', []),
    MSG_COLORIZATION_ENABLE : TAMDescriptor('MSG_COLORIZATION_ENABLE', '>H', ['enable']),
    MSG_VIDEO_ORIENTATION   : TAMDescriptor('MSG_VIDEO_ORIENTATION', '>H', ['value']),
    MSG_VIDEO_SOURCE        : TAMDescriptor('MSG_VIDEO_SOURCE', '>H', ['value']),
    MSG_ICE_ENABLE          : TAMDescriptor('MSG_ICE_ENABLE', '>H', ['value']),
    MSG_ICE_MINMAX          : TAMDescriptor('MSG_ICE_MINMAX', '>H', ['value']),
    MSG_ICE_STRENGTH        : TAMDescriptor('MSG_ICE_STRENGTH', '>H', ['value']),
    MSG_ICE_FREQ_THRESH     : TAMDescriptor('MSG_ICE_FREQ_THRESH', '>H', ['value']),
    MSG_AGC_MODE            : TAMDescriptor('MSG_AGC_MODE', '>H', ['value']),
    MSG_AGC_GAIN            : TAMDescriptor('MSG_AGC_GAIN', '>H', ['value']),
    MSG_AGC_GAIN_BIAS       : TAMDescriptor('MSG_AGC_GAIN_BIAS', '>H', ['value']),
    MSG_AGC_LEVEL_BIAS      : TAMDescriptor('MSG_AGC_LEVEL_BIAS', '>H', ['value']),
    MSG_AGC_GAIN_LIMIT      : TAMDescriptor('MSG_AGC_GAIN_LIMIT', '>H', ['value']),
    MSG_AGC_GAIN_FLATTEN    : TAMDescriptor('MSG_AGC_GAIN_FLATTEN', '>H', ['value']),
    MSG_BLACK_HOT           : TAMDescriptor('MSG_BLACK_HOT', '', ['']),
    MSG_WHITE_HOT           : TAMDescriptor('MSG_WHITE_HOT', '', ['']),
    MSG_TCOMP_STATUS        : TAMDescriptor('MSG_TCOMP_STATUS', '>BBHHHHHH',
                                            ['nranges', 'flags', 'range', 'min',
                                             'max', 'tcomp', 'reserved1', 'reserved2'])
}


class TAMMessage:
    '''TAM message class - holds a TAM protocol message'''
    def __init__(self):
        self._buf = ""
        self._fields = {}
        self._unpacked = False
        self.debug_level = 0

    def __str__(self):
	'''format a message as a string'''
        if not self.valid():
            return 'TAMMessage(INVALID)'
        type = self.msg_type()
        if type in msg_types:
            return msg_types[type].format(self)
        return 'TAMMessage(UNKNOWN %s, %u)' % (str(type), self.msg_length())

    def __getattr__(self, name):
        '''allow access to message fields'''
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        '''allow access to message fields'''
        if name.startswith('_'):
            self.__dict__[name] = value
        else:
            self._fields[name] = value

    def have_field(self, name):
        '''return True if a message contains the given field'''
        return name in self._fields

    def debug(self, level, msg):
        '''write a debug message'''
        if self.debug_level >= level:
            print(msg)

    def unpack(self):
	'''unpack a message'''
        if not self.valid():
            raise TAMError('INVALID MESSAGE')
        type = self.msg_type()
        if not type in msg_types:
            raise TAMError('Unknown message %s length=%u' % (str(type), len(self._buf)))
        msg_types[type].unpack(self)

    def pack(self):
	'''pack a message'''
        if not self.valid():
            raise TAMError('INVALID MESSAGE')
        type = self.msg_type()
        if not type in msg_types:
            raise TAMError('Unknown message %s' % str(type))
        msg_types[type].pack(self)

    def name(self):
	'''return the short string name for a message'''
        if not self.valid():
            raise TAMError('INVALID MESSAGE')
        type = self.msg_type()
        if not type in msg_types:
            raise TAMError('Unknown message %s length=%u' % (str(type), len(self._buf)))
        return msg_types[type].name

    def msg_id(self):
	'''return the message id within the class'''
        return ord(self._buf[1])

    def msg_type(self):
	'''return the message type'''
        return self.msg_id()

    def msg_length(self):
	'''return the payload length'''
        (payload_length,) = struct.unpack('>B', self._buf[2:3])
        return payload_length

    def valid_so_far(self):
	'''check if the message is valid so far'''
        if len(self._buf) > 0 and ord(self._buf[0]) != PREAMBLE1:
            return False
        if self.needed_bytes() == 0 and not self.valid():
            return False
        return True

    def add(self, bytes):
	'''add some bytes to a message'''
        self._buf += bytes
        while not self.valid_so_far() and len(self._buf) > 0:
	    '''handle corrupted streams'''
            self._buf = self._buf[1:]
        if self.needed_bytes() < 0:
            self._buf = ""

    def checksum(self, data=None):
	'''return a checksum tuple for a message'''
        if data is None:
            data = self._buf[:]
        cs = 0
        length = self.msg_length() + 3
        if len(data) < length:
            raise TAMError("%s invalid checksum length")
        for i in range(length):
            cs -= ord(data[i])
            cs &= 0xFF
        return cs

    def valid_checksum(self):
	'''check if the checksum is OK'''
        cs = self.checksum()
        cs2 = ord(self._buf[-1])
        return cs == cs2

    def needed_bytes(self):
        '''return number of bytes still needed'''
        if len(self._buf) < 3:
            return 4 - len(self._buf)
        return self.msg_length() + 4 - len(self._buf)

    def valid(self):
	'''check if a message is valid'''
        return len(self._buf) >= 4 and self.needed_bytes() == 0 and self.valid_checksum()


class TAMSerial:
    '''main TAM serial control class.

    port can be a file (for reading only) or a serial device
    '''
    def __init__(self, port, baudrate=57600, timeout=0):

        self.serial_device = port
        self.baudrate = baudrate
        self.use_sendrecv = False
        self.read_only = False
        self.debug_level = 0

        if self.serial_device.startswith("tcp:"):
            import socket
            a = self.serial_device.split(':')
            destination_addr = (a[1], int(a[2]))
            self.dev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dev.connect(destination_addr)
            self.dev.setblocking(1)
            self.dev.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)            
            self.use_sendrecv = True
        elif os.path.isfile(self.serial_device):
            self.read_only = True
            self.dev = open(self.serial_device, mode='rb')
        else:
            import serial
            self.dev = serial.Serial(self.serial_device, baudrate=self.baudrate,
                                     dsrdtr=False, rtscts=False, xonxoff=False, timeout=timeout)
        self.logfile = None
        self.log = None

    def close(self):
	'''close the device'''
        self.dev.close()
	self.dev = None

    def set_debug(self, debug_level):
        '''set debug level'''
        self.debug_level = debug_level

    def debug(self, level, msg):
        '''write a debug message'''
        if self.debug_level >= level:
            print(msg)

    def set_logfile(self, logfile, append=False):
	'''setup logging to a file'''
        if self.log is not None:
            self.log.close()
            self.log = None
        self.logfile = logfile
        if self.logfile is not None:
            if append:
                mode = 'ab'
            else:
                mode = 'wb'
            self.log = open(self.logfile, mode=mode)

    def write(self, buf):
        '''write some bytes'''
        if not self.read_only:
            if self.use_sendrecv:
                return self.dev.send(buf)
            return self.dev.write(buf)

    def read(self, n):
        '''read some bytes'''
        if self.use_sendrecv:
            import socket
            try:
                return self.dev.recv(n)
            except socket.error as e:
                return ''
        return self.dev.read(n)

    def seek_percent(self, pct):
	'''seek to the given percentage of a file'''
	self.dev.seek(0, 2)
	filesize = self.dev.tell()
	self.dev.seek(pct*0.01*filesize)

    def receive_message(self, ignore_eof=False):
	'''blocking receive of one TAM message'''
        msg = TAMMessage()
        while True:
            n = msg.needed_bytes()
            b = self.read(n)
            if not b:
                if ignore_eof:
                    time.sleep(0.01)
                    continue
                return None
            msg.add(b)
            if self.log is not None:
                self.log.write(b)
                self.log.flush()
            if msg.valid():
                return msg

    def receive_message_noerror(self, ignore_eof=False):
	'''blocking receive of one TAM message, ignoring errors'''
        try:
            return self.receive_message(ignore_eof=ignore_eof)
        except TAMError as e:
            print(e)
            return None
        except OSError as e:
            # Occasionally we get hit with 'resource temporarily unavailable'
            # messages here on the serial device, catch them too.
            print(e)
            return None

    def send(self, msg):
	'''send a preformatted TAM message'''
        if not msg.valid():
            self.debug(1, "invalid send")
            return
        if not self.read_only:
            self.write(msg._buf)        

    def send_message(self, msg_id, payload=''):
	'''send a TAM message with class, id and payload'''
        msg = TAMMessage()
        msg._buf = struct.pack('>BBB', PREAMBLE1, msg_id, len(payload))
        msg._buf += payload
        cs = msg.checksum(msg._buf[:])
        msg._buf += struct.pack('>B', cs)
        try:
            print(msg)
        except Exception:
            pass
        self.send(msg)

    def send_uint16(self, msg_id, value):
        '''send a uint16 command'''
        self.send_message(msg_id, struct.pack('>H', value))

    def get_tcomp(self):
        '''send a uint16 command'''
        self.send_message(MSG_TCOMP_STATUS)
        msg = self.receive_message()
        if msg is None:
            return None
        if msg.msg_id() != MSG_TCOMP_STATUS:
            return None
        return msg
    
    def wait_ack(self):
        '''wait for message ack'''
        while True:
            msg = self.receive_message()
            if msg is None:
                break
            print(str(msg))
            sys.stdout.flush()
            if msg.msg_id() in [MSG_ACK, MSG_NACK, MSG_ERR]:
                break
        

########################################################
# EXAMPLE MAIN PROGRAM
########################################################
if __name__ == '__main__':
    from optparse import OptionParser
    
    parser = OptionParser("tamarisk_serial.py [options]")
    parser.add_option("--port", help="serial port", default='/dev/ttyACM0')
    parser.add_option("--baudrate", type='int',
                      help="serial baud rate", default=57600)
    parser.add_option("--video-orientation", type='int', default=VID_ORIENT_NORMAL)
    parser.add_option("--video-source", type='int', default=VID_SRC_AGC)
    parser.add_option("--agc-mode", type='int', default=AGC_MODE_AUTO)
    parser.add_option("--ice-enable", type='int', default=1)
    parser.add_option("--ice-strength", type='int', default=3)
    parser.add_option("--white-hot", type='int', default=1)
    parser.add_option("--colorization-enable", type='int', default=0)

    (opts, args) = parser.parse_args()

    dev = TAMSerial(opts.port, baudrate=opts.baudrate, timeout=2)

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
        
