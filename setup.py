from setuptools import setup, Extension
import numpy as np
import platform, os, sys

version = '0.1.0'

ext_modules = []

extra_compile_args=["-std=gnu99", "-O3"]

paths = ['/home/tridge/project/UAV/prefix',
         '/home/falcon/prefix',
         '/usr/local']
aravis_inst_path = None
for p in paths:
    if os.path.exists(os.path.join(p, 'include/aravis-0.4')):
        aravis_inst_path = p
        break

if aravis_inst_path is None:
    print("Failed to find aravis libraries")
    sys.exit(1)

tamarisk = Extension('PyTamarisk.tamarisk.tamarisk',
                     sources = ['PyTamarisk/tamarisk/tamarisk_py.c'],
                     libraries = ['aravis-0.4'],
                     library_dirs = ['%s/lib' % aravis_inst_path],
                     extra_compile_args=extra_compile_args + ['-O0', '-I%s/include/aravis-0.4' % aravis_inst_path,
                                                              '-I/usr/include/glib-2.0',
                                                              '-I/usr/lib/x86_64-linux-gnu/glib-2.0/include',
                                                              '-I/usr/lib/arm-linux-gnueabihf/glib-2.0/include',
                                                              '-L%s/lib' % aravis_inst_path])
ext_modules.append(tamarisk)
 
setup (name = 'PyTamarisk',
       zip_safe=True,
       version = version,
       description = 'Tamarisk Camera Capture',
       long_description = '''A python interface to the Tamarisk thermal camera, using the avaris libraries for attaching to the camera over ethernet via a pleora interface board''',
       url = 'https://github.com/tridge/PyTamarisk',
       author = 'Andrew Tridgell',
       author_email = 'andrew-cuav@tridgell.net',
       classifiers=['Development Status :: 4 - Beta',
                    'Environment :: Console',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                    'Operating System :: OS Independent',
                    'Programming Language :: Python :: 2.7',
                    'Topic :: Scientific/Engineering'
                    ],
       license='GPLv3',
       packages = ['PyTamarisk', 'PyTamarisk.tamarisk'],
       scripts = [ 'PyTamarisk/tools/tam_capture.py' ],
       ext_modules = ext_modules)
