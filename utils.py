from dataclasses import dataclass, field
from typing import Any
import itertools

from heapq import heappush, heappop, heapify
from docplex.mp.model import Model


class PrioritizedQueue:
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = float('inf')          # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_elem(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_elem(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        try:
            entry = self.entry_finder.pop(task)
            entry[-1] = self.REMOVED
        except KeyError:
            return
    
    def heappify(self, list_):
        self.pq = list_
        self.entry_finder = {entity[1]: entity for entity in list_}
        heapify(self.pq)

    def pop(self):
        'Remove and return the lowest priority task. Return None if empty.'
        while self.pq:
            priority, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return - priority, task
        return


class OptimizedModel(Model):
    """Modified docplex.mp.model.Model
    The idea applied from oficial cplex repo"""

    def __init__(self, name):
        super().__init__(name=name)
        self._batch_add_constr = []
        self._batch_remove_constr = []

    def add_constraint_batch(self, constr):
        """Add constraints to batch"""
        self._batch_add_constr.append(constr)
        return constr

    def remove_constraint_batch(self, constr):
        """Remove constraints from batch"""
        self._batch_remove_constr.append(constr)

    def solve(self):
        """Solve model adding all batched
        constraints before computing"""
        if self._batch_add_constr:
            super().add_constraints(self._batch_add_constr)
            self._batch_add_constr = []

        if self._batch_remove_constr:
            super().remove_constraints(self._batch_remove_constr)
            self._batch_remove_constr = []

        return super().solve()
