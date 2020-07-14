import pytups.superdict as sd
import pytups.tuplist as tl

import solvers.heuristics as heur
import solvers.heuristics_maintfirst as heur_maint

import package.solution as sol
import package.instance as inst

import data.data_input as di
import time
import logging as log
import os


class SlidingScale(heur.GreedyByMission):

    # def __init__(self, instance, solution=None):
    #
    #     heur.GreedyByMission.__init__(self, instance, solution)
    #     pass

    def solve(self):
        periods = self.instance.get_periods()
        resources = self.instance.get_resources('initial')
        fixed = self.instance.get_fixed_states()
        tasks = self.instance.get_tasks()

        self.check_solution()

        pass

