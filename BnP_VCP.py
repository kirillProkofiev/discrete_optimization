import argparse
import sys
import time
import random
from math import floor, ceil
import collections
from itertools import combinations
import os.path as osp

import networkx as nx
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import dwave_networkx as dnx
from networkx.algorithms import approximation
import numpy as np

from utils import PrioritizedQueue, OptimizedModel

class MaxCliqueBnC:
    def __init__(self, G, verbose=False):
        self.G = G
        self.verbose = verbose
        self.nodes = [v for v in sorted(G.nodes())]
        # constructing master model
        self.master_model = OptimizedModel(name='Vertex coloring problem')
        # color graph and init vars related to quantity of the obtained color sets
        self.ind_sets, self.f_opt, self.x_opt = self.generate_vars()
        print('initial independent sets quantity: ', len(self.ind_sets))
        print('initial solution: ', self.f_opt)
        self.x = {i: self.master_model.continuous_var(name='x_{0}'.format(i)) for i, set_ in sorted(self.ind_sets.items())}
        self.forbidden_set = set() # store forbidden set C
        # generate initial constraints for master problem
        constr = []
        # set names to easily remove and replace them throughout solving slave problem
        names = []
        for v in self.nodes:
            names.append(str(v))
            list_constr = [self.x[i] for i, set_ in sorted(self.ind_sets.items()) if v in set_]
            constr.append(self.master_model.sum(list_constr) >= 1)
        self.master_model.add_constraints(constr, names=names)
        self.master_model.minimize(self.master_model.sum(v for v in self.x.values()))

        # constracting slave model
        self.slave_model = OptimizedModel(name='slave model')
        self.slave_vars = {v: self.slave_model.binary_var(name='y_{0}'.format(v)) for v in self.nodes}
        constr = []
        # generate constraints for the slave model
        # for every adge only one vertex could be in an independent set
        for edge in G.edges():
            v, w = edge
            constr.append(self.slave_vars[v] + self.slave_vars[w] <= 1)
        
        # cliques = self.random_qlique_heuristic()
        # for K in cliques:
        #     constr.append(self.slave_model.sum([self.slave_vars[v] for v in K]) <= 1)

        self.slave_model.add_constraints(constr)
        self.add_clique_constraints()

    def add_clique_constraints(self):
        comp_G = nx.complement(self.G)
        constr = set()
        strategies = ['largest_first', 'smallest_last', 'independent_set',
                      'connected_sequential', 'saturation_largest_first']

        for strategy in strategies:
            independency_dict = self.coloring(strategy, comp_G)
            # strategy adding constraints united by with just one color isn't help to boost
            # alghoritm. Need to create stronger constraints.
            extended_ind_set = self.create_larger_ind_sets(independency_dict)
            constr.update({self.slave_model.sum([self.slave_vars[i] for i in set_]) <= 1 
                            for set_ in list(extended_ind_set.values()) if len(set_) > 1})
        self.slave_model.add_constraints(constr)

    @func_set_timeout(7200)
    def bnp(self):
        # solve with cplex help
        # get float solution
        cps = self.master_model.solve()

        # if solution isn't found - drop this branch
        if cps is None:
            return

        # solve with cplex help
        x = np.array(cps.get_all_values())
        z = cps.get_objective_value()

        # get all constraints from the model related to coloring sets
        all_constraints = [self.master_model.get_constraint_by_name(str(v)) for v in self.nodes]
        x_dual = self.master_model.dual_values(all_constraints)
        if self.verbose:
            print(x, z, self.f_opt)

        # trying to cut branch if it's possible
        for exact in [True, True]:
            can_prune_branch, x, x_dual, z = self.column_generation_loop(x=x, x_dual=x_dual, objective=z, 
                                                            timelimit=1000, exact=exact)
            if can_prune_branch:
                return
        # choose branch
        i = self.branching_int(x)
        if  i == -1:
            can_prune_branch, x, x_dual, z = self.column_generation_loop(x=x, x_dual=x_dual, objective=z, 
                                        timelimit=float('inf'), exact=True)
            # since we come to new solution need to check idx again
            i = self.branching_int(x)
            if  i == -1:
                if  z < self.f_opt:
                    self.f_opt = z 
                    self.x_opt = x
                    print(f'new solution: {z}')
                return
        # choose closest to int branch
        seq_br = [0,1] if x[i] < 0.5 else [1,0]
        for b in seq_br:
            branch_constraint = self.master_model.add_constraint_bath(self.x[i] == b)

            if b == 0:
                # get new forbidden set and put it to constraints for slave model
                forb_set = self.ind_sets[i]
                self.forbidden_set.add(frozenset(forb_set))
                forb_constr = self.slave_model.add_constraint_bath(self.slave_model.sum([self.slave_vars[i] for i in forb_set]) <= len(forb_set) - 1)

            self.bnp()
            self.master_model.remove_constraint_bath(branch_constraint)
            if b == 0:
                # remove set from forbidden and remove it from slave model constraints
                self.slave_model.remove_constraint_bath(forb_constr)
                if self.ind_sets[i] in self.forbidden_set:
                    self.forbidden_set.remove(self.ind_sets[i])

    def column_generation_loop(self, x, x_dual, objective, timelimit=1000, exact=False):
        while True:
            col, UBsp = self.column_generation(x_dual, timelimit, exact)
            if not col:
                LB = self.trunc_up(objective / UBsp)
                if LB >= self.f_opt:
                    return True, x, x_dual, objective
                break

            assert dnx.is_independent_set(self.G, col)
            self.ind_sets[len(self.ind_sets)] = col
            self.update_master_model()
            # solve and get new solution
            cps = self.master_model.solve()
            objective = cps.get_objective_value()
            x = cps.get_all_values()
            all_constraints = [self.master_model.get_constraint_by_name(str(v)) for v in self.nodes]
            x_dual = self.master_model.dual_values(all_constraints)

        return False, x, x_dual, objective
    
    def update_master_model(self):
        ''' finction to update constraints, main variables and objective in the model '''
        self.x.update({len(self.x): self.master_model.continuous_var(name='x_{0}'.format(len(self.x)))})
        new_set = self.ind_sets[len(self.ind_sets)-1]
        new_constr = []
        constr = []
        names = []
        for v in new_set:
            names.append(str(v))
            constr = self.master_model.get_constraint_by_name(str(v)).lhs + self.x[len(self.x)-1] >= 1
            new_constr.append(constr)

        old_constraints = [self.master_model.get_constraint_by_name(str(v)) for v in new_set]
        self.master_model.remove_constraints(old_constraints)
        self.master_model.add_constraints(new_constr, names=names)
        self.master_model.apply_batch()
        # update objective
        self.master_model.remove_objective()
        self.master_model.minimize(self.master_model.sum([v for v in self.x.values()]))

    def column_generation(self, x_duals, timelimit, exact):
        ''' gridy separation based on coloring and exact solution'''
        if not exact:
            solution = set()
            weighted_sum = 0
            colors = nx.algorithms.coloring.greedy_color(self.G, strategy='random_sequential')
            # devide weights by the color to get smart weights
            weights = [w / (colors[self.nodes[i]] + 1e-4) for i, w in enumerate(x_duals) ]
            # create prioritized queue
            q = PrioritizedQueue()
            q.heappify([[-w, self.nodes[i]] for i, w in enumerate(weights)])
            next_v = q.pop()
            while next_v:
                weighted_sum += x_duals[next_v[1]]
                solution.add(next_v[1])
                for neigh in self.G.neighbors(next_v[1]):
                    q.remove_elem(neigh)
                next_v = q.pop()
            if weighted_sum > 1 and solution not in self.forbidden_set:
                return solution, 100500
            return None, 100500
        else:
            assert exact
            if timelimit < float('inf'):
                self.slave_model.parameters.timelimit = timelimit
            else:
                self.slave_model.parameters.reset_all()
            self.slave_model.remove_objective()
            self.slave_model.maximize(self.slave_model.sum([(w + 1e-6)*self.slave_vars[i] if w == 0 else w*self.slave_vars[i] 
                                                            for i, w in enumerate(x_duals)])) 
            # self.slave_model.maximize(self.slave_model.sum([w*self.slave_vars[i] for i, w in enumerate(x_duals)]))
            cps = self.slave_model.solve()
            weights = [w for w, v in zip(x_duals, cps.get_all_values()) if v == 1]
            solution = {v for v, w in zip(self.nodes, cps.get_all_values()) if w == 1}
            ub = round(sum(weights), 6)
            if ub > 1:
                assert solution not in self.forbidden_set
                return solution, ub
            return None, 1

    def generate_vars(self):
        ind_sets = set()
        f_opt = 100500
        x = []
        strategies = ['independent_set', 'largest_first', 'smallest_last',
                      'connected_sequential', 'saturation_largest_first']

        for strategy in strategies:
            independency_dict = self.coloring(strategy, G = self.G)
            # strategy adding constraints united by with just one color isn't help to boost
            # alghoritm. Need to create stronger constraints.
            extended_ind_set = self.create_larger_ind_sets(independency_dict)
            for i, set_ in extended_ind_set.items():
                ind_sets.add(frozenset(set_))

            if len(extended_ind_set) < f_opt:
                f_opt = len(extended_ind_set)
                x_opt = extended_ind_set

        for i in range(len(self.nodes)):
            random_color_dict = self.coloring(strategy="random_sequential", G = self.G)
            extended_ind_set_2 = self.create_larger_ind_sets(random_color_dict)
            for i, set_ in extended_ind_set_2.items():
                ind_sets.add(frozenset(set_))

            if len(extended_ind_set) < f_opt:
                f_opt = len(extended_ind_set)
                x_opt = extended_ind_set

        ind_sets = dict(zip(range(len(ind_sets)), ind_sets))
        return ind_sets, f_opt, x_opt

    def random_qlique_heuristic(self):
        ''' slow but better random heuristic '''
        cliques = set()
        for _ in range(300):
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
                    # get vertex with random neighbour
                    random_neigh = random.choice(list(neighbors))
                    C.update({random_neigh})
                cliques.add(frozenset(C))

        return cliques

    def branching_int(self, x: np.array):
        '''function to return vertex with the most small distance to the integer'''
        integer_distances = self.integer_dist_zero(x)
        filtered_vars = integer_distances[integer_distances >= 1e-8]
        if filtered_vars.size == 0:
            return -1
        # find min value of distances
        min_dist_val = np.min(filtered_vars)
        # get an index of the most closest to integer value
        # indexes starts with 1
        i = np.random.choice(np.argwhere(integer_distances == min_dist_val).reshape(-1))
        return i

    def coloring(self, strategy, G):
        ''' color the graph and return reversed dict for computational convenience'''
        colors = nx.algorithms.coloring.greedy_color(G, strategy=strategy)
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
        over a dict with colors and if some vertex from different color has no edge
        with all vertexes in the curent color, we add it to the set.
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
    def integer_dist_zero(x: np.array):
        ''' function to calculate L1 norm between float array and 
        its integer projection '''
        round_func = np.vectorize(round)
        abs_func = np.vectorize(abs)
        integer_distances = abs_func(x - round_func(x))
        return integer_distances

    @staticmethod
    def trunk_down(float_number):
        """wisely floor float number"""
        if np.isclose(float_number, round(float_number), atol=1e-6):
            truncated_number = round(float_number)
        else:
            truncated_number = floor(float_number)
        return truncated_number
    
    @staticmethod
    def trunc_up(float_number):
        """wisely floor float number"""
        if np.isclose(float_number, round(float_number), atol=1e-6):
            truncated_number = round(float_number)
        else:
            truncated_number = ceil(float_number)
        return truncated_number

def main():
    # parse file
    parser = argparse.ArgumentParser(description='max clique problem')
    parser.add_argument('--files_list', '-fl', type=str, default=['jean.col', 'miles750.col', 'miles1000.col','miles1500.col', 'myciel3.col', 'myciel4.col',
                                                                    'queen5_5.col', 'queen6_6.col', 'queen7_7.col'], nargs='*', 
                        help='specify graphs that do you want to compute')
    parser.add_argument('--folder','-fol', type=str, default='DIMACS_bnp', 
                        help='path to the folder where graphs placed')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    inputs_ = [osp.join(args.folder, x) for x in args.files_list]
    # create graph
    for input_ in inputs_:

        G = nx.Graph()
        try:
            with open(input_) as f:
                str_edges = [l[1:].strip().split() for l in f]
            edges = [(int(v2)-1,int(v1)-1) for v1,v2 in str_edges]
            G.add_edges_from(edges)
            max_v = max(G.nodes()) + 1
            assert max_v == len(G.nodes())
        except AssertionError:
            print('\n\n', f'name: {input_}\n','ERROR, corrupted vertexes. There are no consistency in sequence\n\n')
        except:
            print('\n\n', f'name: {input_}\n', 'ERROR in file formating. Please remove meta head from file. Remain lines with edges only\n\n')
        else:
            print('number_of_edges:', G.number_of_edges())
            print('number_of_nodes:', G.number_of_nodes())
            bnp = MaxCliqueBnC(G, verbose=args.verbose)
            start_time = time.time()
            print('--------------start computing--------------')
            try:
                bnp.bnp()
            except FunctionTimedOut:
                print('Watch out! Time out!')
                continue
            finally:
                str_ = (f'name: {input_}\n\nbnb.f_opt {bnp.f_opt}, bnb.x_opt {bnp.x_opt}'
                            + f"\n--- {time.time() - start_time} seconds ---\n\n")
                print(str_)
                with open('test_bnp.txt', 'a') as f:
                    str_ = (f'name: {input_}\n\nbnb.f_opt {bnp.f_opt}, bnb.x_opt {bnp.x_opt}'
                            + f"\n--- {time.time() - start_time} seconds ---\n\n")
                    f.write(str_)

if __name__ == "__main__":
    main()