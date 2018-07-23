# Python

Here all the code related to python technology will be included.

For example:

* models done in `pulp`.
* data wrangling scripts done with `pandas`.
* applications done with `flask`.

Code is organized as follows:

* **scripts**: python scripts done for data wrangling.
* **package**: core code with models, data proessing and main logic.
* **desktop_app**: PyQt gui for app.

## Get the software

Steps to set up development environment:

Windows:

    git clone git@github.com:pchtsp/ROADEF2018.git
    cd ROADEF2018\python
    python3 -m venv venv
    venv\Scripts\activate
    pip3 install --upgrade pip
    pip3 install -r requirements.txt

Ubuntu:

    git clone git@github.com:pchtsp/ROADEF2018.git
    cd ROADEF2018/python
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

## Requirements

Requirements:

* python >= 3.5
* pip install virtualenv
* git
* R
* gurobipy => install manually.

### Ubuntu:

    sudo apt-get install python3 r-core pip git r-base
    pip install virtualenv --user

### Windows

    choco install python3 git r.project pip -y
    pip install virtualenv --user

## For installing requirements in Windows:

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