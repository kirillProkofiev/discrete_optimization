''' This is BnB implementation using heuristic with vertexes degrees, coloring graph and usin Cplex optimization with batch constraints'''
import argparse
import sys
import time
from math import floor, ceil
import collections
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout

from docplex.mp.model import Model
import networkx as nx
from networkx.algorithms import approximation
from dataclasses import dataclass
import numpy as np


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


class MaxCliqueBnB:
    def __init__(self, G, verbose=False):
        self.G = G
        self.verbose = verbose
        self.nodes = [v for v in G.nodes()]

        self.cp = OptimizedModel(name='Max clique problem')
        self.x = {i: self.cp.continuous_var(name='x_{0}'.format(i)) for i in self.nodes}
        constr = []
        for i in self.nodes:
            for j in self.nodes:
                if (i,j) not in G.edges() and (j,i) not in G.edges() and i!=j:
                    constr.append(self.x[i] + self.x[j] <= 1)
        
        for v in self.nodes:
            constr.append(self.x[v] <= 1)

        self.cp.add_constraints(constr)
        self.cp.maximize(self.cp.sum(self.x))
        self.f_opt = len(self.heuristic(G))
        print(f'solution found by heuristic: {self.f_opt}')
        self.x_opt = self.heuristic(G)
        # this is for branching
        self.d = nx.algorithms.coloring.greedy_color(self.G, strategy='independent_set')

        # coloring graph and the vertexes with one color will form independence set
        constr = set()
        strategies = ['largest_first', 'smallest_last', 'independent_set', 'connected_sequential_bfs', 
                        'connected_sequential_dfs', 'saturation_largest_first']
        for strategy in strategies:
            independency_dict = self.coloring(strategy)
            # strategy adding constraints united by with just one color isn't help to boost
            # alghoritm. Need to create stronger constraints.
            extended_ind_set = self.create_larger_ind_sets(independency_dict)
            constr.update({self.cp.sum([self.x[i] for i in set_]) <= 1 for set_ in list(extended_ind_set.values()) if len(set_) != 1})
            
        # heuristic to find approximal independence set
        ind_set_heur = approximation.maximum_independent_set(G)
        constr.update({self.cp.sum([self.x[i] for i in ind_set_heur]) <= 1})

        # now we generate random independence sets proportionately to the quantity of nodes
        # create dict with these sets
        # apply function which extend each independent set
        for i in range(len(self.nodes)):
            random_color_dict = self.coloring('random_sequential')
            extended_ind_set_2 = self.create_larger_ind_sets(random_color_dict)
            constr.update({self.cp.sum([self.x[i] for i in set_]) <= 1 for set_ in list(extended_ind_set_2.values())[:7] if len(set_) != 1})

        # Add all constraints
        print(f'will be added {len(constr)} aditional constraints')
        time.sleep(2)
        self.cp.add_constraints(constr)
        if self.verbose:
            print('_____welcome to matrix____')
            time.sleep(2)

    @func_set_timeout(5)
    def bnb(self):
        # get float solution
        cps = self.cp.solve()

        # if solution isn't found - drop this branch
        if cps is None:
            return

        # solve with cplex help
        x = np.array(cps.get_all_values())
        z = cps.get_objective_value()
        if self.verbose:
            print(x, z, self.f_opt)

        # if our optimal int solution is better than float solution on this branch -> stop
        if self.trunc(z) <= self.f_opt:
            return

        # if all vars are integer (and we do not reduce solution), then rewrite optimal
        # get a branch
        i = self.branching_int(x)
        if i == -1: # if i == -1 than we have all integer vars
            print(f'new solution: {z}')
            self.f_opt = z
            self.x_opt = x
            return

        # go to the left branches
        constraint_left = self.cp.add_constraint_batch(self.x[self.nodes[i]] == 0)
        self.bnb()

        # go to parent node
        self.cp.remove_constraint_batch(constraint_left)

        # go to the right branches
        constraint_right = self.cp.add_constraint_batch(self.x[self.nodes[i]] == 1)
        self.bnb()
        self.cp.remove_constraint_batch(constraint_right)

    def branching_int(self, x: np.array):
        integer_distances = self.integer_dist(x)
        filtered_vars = integer_distances[integer_distances >= 1e-8]
        if filtered_vars.size == 0:
            return -1
        # find min value of distances
        min_dist_val = np.min(filtered_vars)
        # get an index of the most closest to integer value
        # indexes starts with 1
        i = np.random.choice(np.argwhere(integer_distances == min_dist_val).reshape(-1))
        return i
    
    def branching_color(self, x: np.array):
        ''' func to return vertex with largest color for branching '''
        # run through reversive sorted dict of the colors
        for v, c in sorted(self.d.items(), key=lambda x: x[1], reverse=True):
            # if our vertex is not integer then return branch with
            # most big color 
            if abs(x[v-1]-round(x[v-1])) >= 1e-6: # if not int
                return v - 1
        return -1

    def coloring(self, strategy):
        colors = nx.algorithms.coloring.greedy_color(self.G, strategy=strategy)
        # reverse dict
        color_dict = dict()
        for v, c in colors.items():
            if c in color_dict:
                color_dict[c].add(v)
            else:
                color_dict[c] = set([v])
        return color_dict

    def create_larger_ind_sets(self, color_dict : dict):
        """function to create larger independence set simply iterating
        over dict with colors and if some vertex from different color has no edge
        with all vertexes in curent color, we add it to set.
        """
        for color1 in color_dict:
            for color2, vertexes in color_dict.items():
                if color1 != color2:
                    for w in vertexes:
                        if not any(map(lambda x: self.G.has_edge(w,x), color_dict[color1])):
                            color_dict[color1].add(w)
        color_dict = sorted(color_dict.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_dict = collections.OrderedDict(color_dict)
        return sorted_dict

    @staticmethod
    def integer_dist(x: np.array):
        ''' function to calculate L1 norm beetwen float array and 
        its integer projection '''
        round_func = np.vectorize(round)
        abs_func = np.vectorize(abs)
        integer_distances = abs_func(x - round_func(x))
        return integer_distances

    @staticmethod
    def trunc(float_number):
        """wisely floor float number"""
        if np.isclose(float_number, round(float_number), atol=1e-6):
            truncated_number = round(float_number)
        else:
            truncated_number = floor(float_number)
        return truncated_number

    def heuristic(self, G):
        ''' fast heuristic with largest degrees '''
        best_clique = set()
        for v in self.G.nodes():
            C = set([v])
            neighbors = set(self.G.neighbors(v))
            while neighbors:
                for w in C:
                    neighbors.intersection_update(set(self.G.neighbors(w)))
                neighbors -= C # exclude vertexes that are already in clique
                if not neighbors:
                    break
                if len(neighbors)==1:
                    C.update(neighbors)
                    break
                # get vertex with max degree
                deg_list = np.array(self.G.degree(neighbors))
                i = np.argmax(deg_list[:,1])
                max_deg_v = deg_list[i][0]
                C.update({max_deg_v})
        
            if len(C) > len(best_clique):
                best_clique = C
            elif len(C) == len(best_clique):
                # choose one with the largest sum of degrees
                best_clique = (C if sum(np.array(self.G.degree(C))[:,1]) > sum(np.array(self.G.degree(best_clique))[:,1])
                                else best_clique)
        return best_clique

def main():
    # parse file
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--file', '-f', type=str, default='DIMACS_all_ascii\\c-fat500-1.clq', help='specify path to file with graph')
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()
    # create graph
    G = nx.Graph()
    for input_ in ['DIMACS_all_ascii\\gen200_p0.9_55.clq', 
                    'DIMACS_all_ascii\\gen400_p0.9_55.clq',
                    'DIMACS_all_ascii\\brock200_2.clq', 
                    'DIMACS_all_ascii\\brock200_3.clq',
                    'DIMACS_all_ascii\\brock200_4.clq', 
                    'DIMACS_all_ascii\\gen400_p0.9_65.clq',
                    'DIMACS_all_ascii\\C250_9.clq',]:

        with open(input_) as f:
            str_edges = [l[1:].strip().split() for l in f]
        edges = [(int(v2),int(v1)) for v1,v2 in str_edges]
        G.add_edges_from(edges)

        print('number_of_edges:', G.number_of_edges())
        print('number_of_nodes:', G.number_of_nodes())

        bnb = MaxCliqueBnB(G, verbose=False)
        start_time = time.time()
        print('___start computing___')
        try:
            bnb.bnb()
        except FunctionTimedOut:
            print('Time out!')
            continue
        finally:
            with open('test.txt', 'a') as f:
                str_ = (f'name: {input_}\n\nbnb.f_opt {bnb.f_opt}, bnb.x_opt {bnb.x_opt}'
                        + f"\n--- {time.time() - start_time} seconds ---\n\n")
                f.write(str_)

if __name__ == "__main__":
    main()
