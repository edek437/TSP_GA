import argparse
import copy
import sys

import traceback


def parse_file(file_name):
    cost_matrix = []
    with open(file_name) as f:
        for line in f:
            cost_mat_line = line.strip().split(" ")
            cost_mat_line = filter(None, cost_mat_line)
            cost_mat_line = map(lambda x: int(x), cost_mat_line)
            cost_matrix.append(copy.deepcopy(cost_mat_line))
    return cost_matrix

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('graph_path', type=str)
        parser.add_argument('--iterations', type=int, default=100)
        parser.add_argument('--population_size', type=int, default=40)
        parser.add_argument('--crosover_probability', type=float, default=1.0)
        parser.add_argument('--mutation_probability', type=float, default=0.5)
        parser.add_argument('--elitism', type=bool, default=True)
        args = parser.parse_args()
        cost_mat = parse_file(args.graph_path)
        print cost_mat

        print "\n------------------------------------------------------------"
        print "                     Results                                "
        print "------------------------------------------------------------"
        # print "\nBest path = %s" % (best_path_vec,)
        # print "\nBest path cost = %s\n" % (best_path_cost,)

    except Exception, e:
        print "exception: " + str(e)
        traceback.print_exc()
