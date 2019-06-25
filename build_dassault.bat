SET target=build_dassault
rem SET branch=multiple_maintenances
SET branch=HEAD

IF NOT EXIST %target% GOTO TARGETEXISTS
rd /s /q %target%
:TARGETEXISTS
mkdir %target%
git archive --format=tar %branch% python | (cd %target% && tar xf -)
git archive --format=tar %branch% R/functions/import_results.R | (cd %target% && tar xf -)
git archive --format=tar %branch% data/template | (cd %target% && tar xf -)

pause