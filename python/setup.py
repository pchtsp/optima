from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.

includefiles = []
# includefiles = [
# ]
excludes = []
excludes = [
    'scipy.optimize'
    , 'scipy.sparse'
    , 'scipy.special'
    , 'scipy.spatial'
    , 'scipy.linalg'
    , 'pandas.tests'
    ,"tkinter"
    ,'PyQt5.Qt',
                        "PyQt5.QtBluetooth",
                        "PyQt5.QtNetwork",
                        "PyQt5.QtNfc",
                        "PyQt5.QtWebChannel",
                        "PyQt5.QtWebEngine",
                        "PyQt5.QtWebEngineCore",
                        "PyQt5.QtWebEngineWidgets",
                        "PyQt5.QtWebKit",
                        "PyQt5.QtWebKitWidgets",
                        "PyQt5.QtWebSockets",
                        "PyQt5.QtSql",
                        "PyQt5.QtNetwork",
                        "PyQt5.QtScript"]

includes = []

buildOptions = dict(
    packages = ['numpy', 'idna.idnadata'], 
    excludes = excludes,
    includes = includes,
    include_files = includefiles)

import sys
base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('desktop_app/app.py', base=base, targetName = 'optima')
]

setup(name='optima',
      version = '1.0',
      description = '',
      options = dict(build_exe = buildOptions),
      executables = executables)
