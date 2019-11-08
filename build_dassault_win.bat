SET target=build_dassault_win
SET script=python/optima.spec
SET icon=python/desktop_app/plane.ico

rem check if venv is present before continuing
call python\venv\Scripts\activate
pyinstaller -y %script% --distpath %target% --icon=%icon%

echo start optima\optima.exe -gui > %target%\runOptima.bat

pause