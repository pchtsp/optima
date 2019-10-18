# -*- mode: python -*-

import sys
import os

block_cipher = None

def get_pandas_path():
    import pandas
    pandas_path = pandas.__path__[0]
    return pandas_path

def get_palettable_path():
    import palettable
    return palettable.__path__[0]

def get_dfply_path():
    import dfply
    return dfply.__path__[0]

def get_plotly_path():
    import plotly
    return plotly.__path__[0]


path_main = os.path.dirname(os.path.abspath(sys.argv[2]))

a = Analysis(['scripts/main.py'],
             pathex=[path_main],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['pandas.tests'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

a.datas += Tree(get_pandas_path(), prefix='pandas', excludes=["*.pyc", "tests"])
a.datas += Tree(get_palettable_path(), prefix='palettable', excludes=["*.pyc"])
a.datas += Tree(get_dfply_path(), prefix='dfply', excludes=["*.pyc"])
a.datas += Tree(get_plotly_path(), prefix='plotly', excludes=["*.pyc"])
a.datas += Tree('data/template/', prefix='examples', excludes=["*.html", "*/*_files"])
#a.datas += [('R/functions/import_results.R', '../R/functions/import_results.R', 'DATA')]
a.datas += [('plane.ico', 'python/desktop_app/plane.ico', 'DATA')]
a.binaries = filter(lambda x: 'pandas' not in x[0], a.binaries)
a.binaries = [x for x in a.binaries if not x[0].startswith("IPython")]
a.binaries = [x for x in a.binaries if not x[0].startswith("zmq")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("PyQt5")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("Qt5")]
a.binaries = a.binaries - TOC([
 #('sqlite3.dll', None, None),
 ('tcl85.dll', None, None),
 ('tk85.dll', None, None),
 #('_sqlite3', None, None),
 #('_ssl', None, None),
 #('_tkinter', None, None)
 ])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# for several files, keep the following two and comment the other one.
exe = EXE(pyz, a.scripts, exclude_binaries=True, name='optima', debug=False, strip=False, upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name='optima')

# for one exe, replace the two above for.
# exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, name='optima', debug=False, strip=False, upx=True, runtime_tmpdir=None, console=True)

# pyinstaller -y optima.spec