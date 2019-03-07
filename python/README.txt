# Python

## Get the software

Steps to set up development environment:

Ubuntu:

    git clone git@github.com:pchtsp/optima.git
    cd optima/python
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

Windows:

Only difference is using `venv/Scripts/activate` instead of `source venv/bin/activate`.

Note: r packages are automatically installed when trying to run it for the first time.

## Requirements

Requirements:

* python >= 3.6
* R >= 3.5

Recommended:

* git
* gurobipy => install manually.

### Ubuntu:

    sudo apt-get install python3 r-core git r-base

### Windows

    choco install python3 git r.project -y

## For installing requirements in Windows:

Sometimes there are problems with using `pip` directly to install libraries.

I only had to install `numpy` and the build tools 2017.

Check: https://stackoverflow.com/a/32064281

Check: https://wiki.python.org/moin/WindowsCompilers

https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads

https://visualstudio.microsoft.com/es/downloads/?rr=https%3A%2F%2Fwiki.python.org%2Fmoin%2FWindowsCompilers

* Build Tools 2017: http://landinghub.visualstudio.com/visual-cpp-build-tools
* numpy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
* Scipy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
* cx_freeze in github version, not pip.
* specific configuration for windows?

## Examples of using some scripts

`exec_iteratively` and `exec.py` both accept json-styled options. They replace the default parameters.

    python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/pchtsp/Documents/projects/OPTIMA/results/\"}" -d "{\"solver\": \"CPLEX\"}" > log.txt &
    python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/disc/f.peschiera/Documents/projects/optima/results/clust1_20181015/\"}" -d "{\"solver\": \"CPLEX\"}" > log_20181015.txt &
    python3 python/scripts/exec.py -d "{\"solver\": \"GUROBI\"}" > log.txt &

## Example of using the template

If not loaded, the python environment needs to be loaded:

    cd optima/python
    source venv/bin/activate

Then, the command to take a given `template_in.xlsx` file inside the directory `201902141830`. Additionally, an `options_in.json` file can be next to the template file.

    python3 python/scripts -id /home/pchtsp/Documents/projects/OPTIMA/data/template/201902141830/

## Output parameters

The solving process creates several output files. Below a description of the files:

**template_out.xlsx**:  output data following excel template.
**output.log**:  solving process.
**errors.json**:  best solution infeasibilities.
**data_out.json**: complete solution in json format.
**options_out.json**: all options used.
**data_in.json**: input data in json format.
**solution.html**: web gantt produced with the best found solution.

## Building executable

The following commands build the optima.exe:

    cd optima/python
    source venv/bin/activate
    pyinstaller -y optima.spec