''' This is BnC implementation using heuristic with random vertexes choise or max degree, coloring graph and using Cplex optimization with batch constraints'''
import argparse
import sys
import time
import random
from math import floor, ceil
import collections
from itertools import combinations
import os.path as osp
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
from queue import PriorityQueue

import networkx as nx
from networkx.algorithms import approximation
import numpy as np

from utils import PrioritizedQueue, OptimizedModel


class MaxCliqueBnC:
    def __init__(self, G, verbose=False):
        self.G = G
        self.verbose = verbose
        self.nodes = [v for v in G.nodes()]

        self.cp = OptimizedModel(name='Max clique problem')
        self.x = {i: self.cp.continuous_var(name='x_{0}'.format(i)) for i in self.nodes}
        constr = []
        
        for v in self.nodes:
            constr.append(self.x[v] <= 1)

        self.cp.add_constraints(constr)
        self.cp.maximize(self.cp.sum(self.x))
        self.x_opt = self.heuristic(G)
        self.f_opt = len(self.x_opt)
        print(f'solution found by heuristic: {self.f_opt}')

        # this is for the color branching
        self.d = nx.algorithms.coloring.greedy_color(self.G, strategy='independent_set')
        # launch initial constrains
        constr = self.init_coloring(True)
        # Add all constraints
        if constr:
            print(f'will be added {len(constr)} aditional constraints')
            time.sleep(2)
            self.cp.add_constraints(constr)
        else:
            print("coloring is off")

        if self.verbose:
            print('_____welcome to matrix____')
            time.sleep(2)

    @func_set_timeout(7200)
    def bnc(self):
        # solve with cplex help
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

        # if our optimal int solution is better than the float solution on this branch -> stop
        if self.trunc(z) <= self.f_opt:
            return

        while True:
            sep_constr = []
            for _ in range(10):
                C = self.separation(x)
                if C:
                    sep_constr.append(C)
            if not sep_constr:
                break
            for set_cons in sep_constr:
                self.cp.add_constraint_batch(self.cp.sum([self.x[i] for i in set_cons]) <= 1)
            cps2 = self.cp.solve()
            if cps2 is None:
                return
            x2 = np.array(cps2.get_all_values())
            z2 = cps2.get_objective_value()
            if self.trunc(z2) <= self.f_opt:
                return
            if z2 - z < 1e-3: # if changes too small
                break
            z = z2
            x = x2

        # whitening constraints. Delete non-binding constraints
        if self.cp.number_of_constraints > 2000:
            self.delete_non_bidings()

        # get a branch
        i = self.branching_int(x)
        # if all vars are integer (and we do not reduce solution), then rewrite the optimal
        if i == -1: # if i == -1 than we have all integer vars
            # checking if our solution is clique or not
            cor_constr = self.check_clique(x) # maybe add only 1/3
            if cor_constr:
                # if it's not a clique, add constraints for no edge nodes
                cor_constr = np.random.permutation(list(cor_constr))[:int(len(cor_constr)/2)]
                self.cp.add_constraints(list(cor_constr))
                self.bnc()
            else:
                print(f'new solution: {z}')
                self.f_opt = z
                self.x_opt = x
                return

        # choose closer branch
        seq_br = [0,1] if x[i] < 0.5 else [1,0]
        for b in seq_br:
            # go to the branch
            branch_constraint = self.cp.add_constraint_batch(self.x[self.nodes[i]] == b)
            self.bnc()

            # go to parent node
            self.cp.remove_constraint_batch(branch_constraint)

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
    
    def branching_color(self, x: np.array):
        ''' function to return vertex with largest color for branching '''
        # run through reversive sorted dict of the colors
        for v, _ in sorted(self.d.items(), key=lambda x: x[1], reverse=True):
            # if our vertex is not integer then return branch with
            # most big color 
            if abs(x[v-1]-round(x[v-1])) >= 1e-6: # if not int
                return v - 1
        return -1

    def coloring(self, strategy):
        ''' color the graph and return reversed dict for computational convenience'''
        colors = nx.algorithms.coloring.greedy_color(self.G, strategy=strategy)
        # reverse dict
        color_dict = dict()
        for v, c in colors.items():
            if c in color_dict:
                color_dict[c].add(v)
            else:
                color_dict[c] = set([v])
        return color_dict

    def delete_non_bidings(self):
        ''' function to delete non-bindings constraints from the model'''
        # whitening constraints. Delete non-binding constraints
        num_of_constr = self.cp.number_of_constraints
        # get all constraints from the model except first one for vertexes
        all_constraints = [self.cp.get_constraint_by_index(i) for i in
                           range(len(self.nodes), num_of_constr)]
        # filter it by size
        filtered_constr = [x for x in all_constraints if ((x is not None) and (x.lhs.size() > 1))]
        # sort by slack value and delete 2/3 of them
        constr_to_del = sorted(filtered_constr, key=lambda x: x.slack_value, reverse=True)[:int(len(filtered_constr) / 1.5)]
        self.cp.remove_constraints(constr_to_del)
        print(f"delete non-binding constraints from {num_of_constr} --> {self.cp.number_of_constraints}")
    
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
    
    def random_heuristic(self, G):
        ''' slow but better random heuristic '''
        best_clique = set()
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
            
                if len(C) > len(best_clique):
                    best_clique = C
                elif len(C) == len(best_clique):
                    # choose one with the largest sum of degrees
                    best_clique = (C if sum(np.array(self.G.degree(C))[:,1]) > sum(np.array(self.G.degree(best_clique))[:,1])
                                    else best_clique)
        return best_clique

    def stupid_separation(self, x):
        # color graph to get independent set
        colors = self.coloring(strategy = 'random_sequential')
        # extend it
        color_dict = self.create_larger_ind_sets(colors)
        max_weighted_set = self.get_max_weighted_ind_set(color_dict, x)
        return max_weighted_set
    
    def separation(self, x):
        ''' gridy separation based on coloring'''
        solution = set()
        weighted_sum = 0
        colors = nx.algorithms.coloring.greedy_color(self.G, strategy='random_sequential')
        # devide weights by the color to get smart weights 
        weights = [w / (colors[self.nodes[i]] + 1e-4) for i, w in enumerate(x) ]
        # create prioritized queue
        q = PrioritizedQueue()
        q.heappify([[-w, self.nodes[i]] for i, w in enumerate(weights)])
        next_v = q.pop()
        while next_v:
            weighted_sum += x[next_v[1] - 1]
            solution.add(next_v[1])
            for neigh in self.G.neighbors(next_v[1]):
                q.remove_elem(neigh)
            next_v = q.pop()
        if weighted_sum > 1:
            return solution

    @staticmethod
    def get_max_weighted_ind_set(color_dict, weights):
        ''' find max weighted independent set '''
        max_sum = 0
        max_weighted_set = None
        for value in color_dict.values():
            set_sum = [weights[v-1] for v in value]
            cur_sum = np.sum(set_sum)
            if cur_sum > 1 and cur_sum > max_sum:
                max_weighted_set = value
                max_sum = cur_sum
        
        return max_weighted_set

    def check_clique(self, nodes):
        solution = [ind + 1 for ind, v in enumerate(nodes) if v == 1.]
        cor_edges = set()
        #check each possible pair
        for (u,v) in combinations(solution,2):  
            if u != v and not self.G.has_edge(u,v):
                cor_edges.update({self.x[u] + self.x[v] <= 1})
        return cor_edges
    
    def init_coloring(self, to_do_flag=True):
        if not to_do_flag:
            return
        # color the graph and the vertexes with one color will form an independence set
        constr = set()
        strategies = ['largest_first', 'smallest_last', 'independent_set', 'connected_sequential_bfs', 
                        'connected_sequential_dfs', 'saturation_largest_first']

        for strategy in strategies:
            independency_dict = self.coloring(strategy)
            # strategy adding constraints united by with just one color isn't help to boost
            # alghoritm. Need to create stronger constraints.
            extended_ind_set = self.create_larger_ind_sets(independency_dict)
            constr.update({self.cp.sum([self.x[i] for i in set_]) <= 1 
                            for set_ in list(extended_ind_set.values()) if len(set_) != 1})
                
        # heuristic to find approximal independence set
        ind_set_heur = approximation.maximum_independent_set(self.G)
        constr.update({self.cp.sum([self.x[i] for i in ind_set_heur]) <= 1})

        # now we generate random independence sets proportionately to the quantity of the nodes
        # create a dict with these sets
        # apply the function that extend each independent set
        for i in range(len(self.nodes)):
            random_color_dict = self.coloring(strategy="random_sequential")
            extended_ind_set_2 = self.create_larger_ind_sets(random_color_dict)
            constr.update({self.cp.sum([self.x[i] for i in set_]) <= 1 
                          for set_ in list(extended_ind_set_2.values())[:3] if len(set_) != 1})
        return constr

def main():
    # parse file
    parser = argparse.ArgumentParser(description='max clique problem')
    parser.add_argument('--files_list', '-fl', type=str, default=['C125.9.clq'], nargs='*', 
                        help='specify graphs that do you want to compute')
    parser.add_argument('--folder','-fol', type=str, default='DIMACS_all_ascii', 
                        help='path to the folder where graphs placed')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    inputs_ = [osp.join(args.folder, x) for x in args.files_list]
    # create graph
    for input_ in inputs_:

        G = nx.Graph()
        with open(input_) as f:
            str_edges = [l[1:].strip().split() for l in f]
        edges = [(int(v2),int(v1)) for v1,v2 in str_edges]
        G.add_edges_from(edges)

        print('number_of_edges:', G.number_of_edges())
        print('number_of_nodes:', G.number_of_nodes())

        bnc = MaxCliqueBnC(G, verbose=args.verbose)
        start_time = time.time()
        print('--------------start computing--------------')
        try:
            bnc.bnc()
        except FunctionTimedOut:
            print('Watch out! Time out!')
            continue
        finally:
            with open('test.txt', 'a') as f:
                str_ = (f'name: {input_}\n\nbnb.f_opt {bnc.f_opt}, bnb.x_opt {bnc.x_opt}'
                        + f"\n--- {time.time() - start_time} seconds ---\n\n")
                f.write(str_)

if __name__ == "__main__":
    main()
