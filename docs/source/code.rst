Classes
**************************

This document details the main classes and methods used on the planning of a solution. Each class serves a specific purpose.


Instance
=========================

Represents the standardized input data read from one of many sources. Its methods transform this input data for better usage in other functions. It also deals with deprecation and default values.

:py:class:`Instance`

.. automodule:: package.instance
   :members:


Solution
=========================

Represents the standardized output data of the planning decision. Its methods transform this input data for better usage in other functions.

:py:class:`Solution`

.. automodule:: package.solution
   :members:

Experiment
=========================

Represents the combination of an instance and a solution. Its methods check for feasibility, and generate reports of the solution. They also modify a given solution and complete a solution with auxiliary data. It can be initialized from a directory.

:py:class:`Experiment`

.. automodule:: package.experiment
   :members:


GreedyByMission
=========================

A subclass of Experiment that implements a number of methods to generate and modify a given solution. It implements a logic where each mission is included in an ordered way to the planning.

:py:class:`Heuristic`

.. automodule:: solvers.heuristics
   :members:

MaintenanceFirst
=========================

A subclass of Heuristic that implements a simulated annealing algorithm to obtain a solution.

:py:class:`MaintenanceFirst`

.. automodule:: solvers.heuristics_maintfirst
   :members:

MIP model
=========================

A subclass of Experiment that implements a MIP model to solve the mission assignments.

:py:class:`ModelMissions`

.. automodule:: solvers.model_dassault
   :members:
