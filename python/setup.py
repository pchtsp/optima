from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.

includefiles = []
# includefiles = [
# 'package/aux.py'
# ,'package/config.py'
# ,'package/data_input.py'
# ,'package/heuristics.py'
# ,'package/instance.py'
# ,'package/logFiles.py'
# ,'package/model.py'
# ,'package/params.py'
# ,'package/solution.py'
# ,'package/tests.py'
# ,'desktop_app/gui.py'
# ]

includes = []

buildOptions = dict(
    packages = ['numpy', 'sparse'], 
    excludes = [],
    includes = includes,
    include_files = includefiles)

import sys
base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('app.py', base=base, targetName = 'optima')
]

setup(name='optima',
      version = '1.0',
      description = '',
      options = dict(build_exe = buildOptions),
      executables = executables)
