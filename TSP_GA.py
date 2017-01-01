#!/usr/bin/env python
import argparse
import copy
import random
import traceback
from operator import itemgetter
import logging
import sys


class GA(object):
    def __init__(self, args):
        self.args = args
        self.graph = self._parse_file()
        self.population = []
        self.total_fitness = 0
        self._generate_random_population()

    def _parse_file(self):
        cost_matrix = []
        with open(self.args.graph_path) as f:
            for line in f:
                cost_mat_line = line.strip().split(" ")
                cost_mat_line = filter(None, cost_mat_line)
                cost_mat_line = map(lambda x: int(x), cost_mat_line)
                cost_matrix.append(copy.deepcopy(cost_mat_line))
        return cost_matrix

    # generate initial population for GA algorithm
    def _generate_random_population(self):
        pop_size = self.args.population_size
        for i in xrange(0, pop_size):
            chromosome = range(0, len(self.graph))
            random.shuffle(chromosome)
            fitness = self._count_fitness(chromosome)
            self.total_fitness += fitness  # needed for roulette wheel selection
            # TODO: what about identical offsprings??
            self.population.append((fitness, copy.deepcopy(chromosome)))
        self.population = sorted(self.population, key=lambda tup: tup[0])  # needed for rank selection

    # return tuple (fitness, chromosome)
    def _count_fitness(self, chromosome):
        fitness = 0
        start_town_ind = 0
        dest_town_ind = 0
        for ind in xrange(0, len(chromosome)-1):
            start_town_ind = chromosome[ind]
            dest_town_ind = chromosome[ind +1]
            fitness += self.graph[start_town_ind][dest_town_ind]
        return fitness

    def _generate_next_population(self):
        new_population = []
        new_population_fitness = 0
        current_population_size = len(self.population)
        if self.args.elitism:
            # TODO: Fix odd population size problem
            new_population.append(self.population[:min(self.args.elitism_nbr/2*2, current_population_size)]) # ensure even elitism

        while len(new_population) is not current_population_size:
            parents = getattr(self, '_' + self.args.selection_method + '_select')()
            child1 = getattr(self, '_' + self.args.crossover_method + '_crossover')(parents[0]. parents[1])
            child2 = getattr(self, '_' + self.args.crossover_method + '_crossover')(parents[1]. parents[0])
            for child in [child1, child2]:
                self._mutate(child)
                fitness = self._count_fitness(child)
                new_population_fitness += fitness
                # TODO: what about identical offsprings??
                new_population.append((fitness, child))

        self.total_fitness = new_population_fitness
        self.population = sorted(new_population, key=lambda tup: tup[0])

    # return two chromosomes picked by roulette wheel selection
    def _roulette_wheel_select(self):
        parent_chromosomes = []
        for it in [0, 1]:
            selector = random.random()
            checked_chrom_fitness_sum = 0.0
            for member in self.population:
                checked_chrom_fitness_sum += member[0]
                if selector < checked_chrom_fitness_sum/self.total_fitness:
                    parent_chromosomes[it] = member[1]
                    break

        return parent_chromosomes

    # return two chromosomes picked by rank selection
    def _rank_select(self):
        parent_chromosomes = []
        for it in [0, 1]:
            selector = random.random()
            pop_size = len(self.population)
            rank_sum = pop_size * (pop_size - 1) * 0.5
            checked_chrom_rank_sum = 0
            for ind in xrange(0, pop_size):
                checked_chrom_rank_sum += (pop_size - ind)
                if selector < checked_chrom_rank_sum/rank_sum:
                    parent_chromosomes[it] = self.population[ind][1]
                    break

        return parent_chromosomes

    # return two offsprint of parent1 and parent2 generated using PMX method
    def _pmx_crossover(self, parent1, parent2):
        if random.random() >= self.args.crossover_probability:
            return parent1

    # return two offsprint of parent1 and parent2 generated using ER method
    def _er_crossover(self, parent1, parent2):
        if random.random() >= self.args.crossover_probability:
            return parent1

        # generate neighbour list
        neighbours_dict = {}
        for node in xrange(0, len(parent1)):
            neighbours = []
            for chromosome in [parent1, parent2]:
                ind = chromosome.index(node)
                neighbours.append(chromosome[(ind+1) % len(chromosome)])
                neighbours.append(chromosome[(ind-1) % len(chromosome)])
            neighbours_dict[(len(neighbours), node)] = neighbours

        # edge recombination
        child = [random.choice([parent1, parent2])[0]]
        while len(child) is not len(parent1):
            for node, neighbours in neighbours_dict.iteritems():
                updated_neighbours = [n for n in neighbours if n != child[-1]]
                neighbours_dict[(len(updated_neighbours), node)] = updated_neighbours
            key_list = neighbours_dict.keys()
            next_node = random.choice([k for k in key_list if k[0] == min(key_list, key=lambda t: t[1])])
            child.append(next_node[1])
            del neighbours_dict[next_node]

        return child

    def _mutate(self, chromosome):
        if random.random() >= self.args.mutation_probability:
            chromosome_len = len(chromosome)
            rand_ind1 = random.randint(0, chromosome_len)
            rand_ind2 = random.randint(0, chromosome_len)
            chromosome[rand_ind1], chromosome[rand_ind2] = chromosome[rand_ind2], chromosome[rand_ind1]

    def run(self):
        print self.graph
        print "-----------------"
        for it in xrange(0, self.args.iterations):
            print self.population
            print "-----------------"
            self._generate_next_population()

        return min(self.population, key=itemgetter(0))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('graph_path', type=str)
        parser.add_argument('--iterations', type=int, default=100)
        parser.add_argument('--population_size', type=int, default=40)
        parser.add_argument('--selection_method', type=str, choices=['roulette_wheel', 'rank'], default='roulette_wheel')
        parser.add_argument('--crossover_method', type=str, choices=['pmx', 'er'], default='pmx')
        parser.add_argument('--crossover_probability', type=float, default=1.0)
        parser.add_argument('--mutation_probability', type=float, default=0.5)
        parser.add_argument('--elitism', type=bool, default=True)
        parser.add_argument('--elitism_nbr', type=int, default=2)

        args = parser.parse_args()
        print "\n------------------------------------------------------------"
        print "                   Running TSP GA for args                  "
        print "------------------------------------------------------------"
        print vars(args)
        # sys.exit(0)
        tsp_ga = GA(args)
        best_path = tsp_ga.run()
        print "\n------------------------------------------------------------"
        print "                     Results                                "
        print "------------------------------------------------------------"
        print "\nBest path = %s" % (best_path[1])
        print "\nBest path cost = %s\n" % (best_path[0])

    except Exception, e:
        print "exception: " + str(e)
        traceback.print_exc()
