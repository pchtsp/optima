SET target=src_dassault
rem SET branch=multiple_maintenances
SET branch=HEAD

IF NOT EXIST %target% GOTO TARGETNOTEXISTS
rd /s /q %target%
:TARGETNOTEXISTS
mkdir %target%
git archive --format=tar %branch% python/data | (cd %target% && tar xf -)
git archive --format=tar %branch% python/desktop_app | (cd %target% && tar xf -)
git archive --format=tar %branch% python/package | (cd %target% && tar xf -)
git archive --format=tar %branch% python/reports | (cd %target% && tar xf -)
git archive --format=tar %branch% python/scripts/main.py | (cd %target% && tar xf -)
git archive --format=tar %branch% python/solvers/heuristics.py | (cd %target% && tar xf -)
git archive --format=tar %branch% python/solvers/heuristics_maintfirst.py | (cd %target% && tar xf -)
git archive --format=tar %branch% python/solvers/model_dassault.py | (cd %target% && tar xf -)
git archive --format=tar %branch% python/requirements.txt | (cd %target% && tar xf -)
git archive --format=tar %branch% python/optima.spec | (cd %target% && tar xf -)
git archive --format=tar %branch% python/tests | (cd %target% && tar xf -)
git archive --format=tar %branch% build_dassault_win.bat | (cd %target% && tar xf -)
git archive --format=tar %branch% data/template | (cd %target% && tar xf -)

rem symbolic link for venv for testing.
rem Xcopy /E /I %cd%\python\venv %cd%\%target%\python\venv

pause