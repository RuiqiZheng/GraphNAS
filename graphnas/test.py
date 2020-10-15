"""Entry point."""

import argparse
import time
import numpy
import torch
import pygad
import graphnas.trainer as trainer
import graphnas.utils.tensor_utils as utils
import ssl
import logging
import random
from graphnas.search_space import MacroSearchSpace

ssl._create_default_https_context = ssl._create_unverified_context
POPULATION_SIZE = 100
search_space = [
["gat", "gcn", "cos", "const", "gat_sym", "linear", "generalized_linear"],
["sum", "meaen", "max", "mlp"],
["sigmoid", "tanh", "relu", "linear", "softplus", "leaky_relu", "relu6", "elu"],
[1, 2, 4, 6, 8, 16],
[4, 8, 16, 32, 64, 128,256],
]


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='genetic_algorithm',
                        choices=['train', 'derive','genetic_algorithm'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--max_save_num', type=int, default=5)
    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=100,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=100)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")


#def main(args):  # pylint:disable=redefined-outer-name

    

def fitness_func(chromosome):
    args = build_args()
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    utils.makedirs(args.dataset)
    trnr = trainer.Trainer(args)

    
    fitness = trnr.genetic_get_reward(chromosome)
    #fitness = trnr.genetic_get_reward([
     #   ['gat', 'sum', 'linear', 4, 128, 'linear', 'sum', 'elu', 8, 6],
      #  ['gcn', 'sum', 'tanh', 6, 64, 'cos', 'sum', 'tanh', 6, 3],
       # ['const', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7],
   # ])
    return fitness


def generate_layer(num_within_layer):
    layer = []
    for i in range(num_within_layer):
        layer.append(random.choice(search_space[0]))
        layer.append(random.choice(search_space[1]))
        layer.append(random.choice(search_space[2]))
        layer.append(random.choice(search_space[3]))
        layer.append(random.choice(search_space[4]))
        
    return layer

class Individual(object):
    '''
    Class representing individual in population
    '''
    def __init__(self, chromosome = []):
        
        if not chromosome:
            self.chromosome = []
            self.num_layer = random.randrange(1,3)
            for i in range(self.num_layer):
                num_within_layer = random.randrange(1,2)
                layer = generate_layer(num_within_layer)
                self.chromosome.append(layer)
        else:
            self.chromosome = chromosome
            self.num_layer = len(chromosome)
        print(sum(self.chromosome,[]))
        self.fitness = fitness_func(sum(self.chromosome,[]))

        print(self.fitness)


    def mutated_genes(self):
       mute_layer = random.randrange(self.num_layer)
       gen_within_layer = random.randrange(len(self.chromosome[mute_layer]))
       return random.choice(search_space[gen_within_layer%5])

    def mate(self, par2):
        '''
        Perform mating and produce new offspring
        '''
# chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
# random probability
            prob = random.random()
# if prob is less than 0.45, insert gene
            # from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)
# if prob is between 0.45 and 0.90, insert
            # gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)
# otherwise insert random gene(mutate),
            # for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())
# create new Individual(offspring) using
        # generated chromosome for offspring
        return Individual(child_chromosome)

# Driver code
def main():
    args = build_args()
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    # args.max_epoch = 1
    # args.controller_max_step = 1
    # args.derive_num_sample = 1
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args)



# current generation
    generation = 1
    found = False
    population = []
# create initial population
    for _ in range(POPULATION_SIZE):
        population.append(Individual())
    while not found:
# sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness)
    # if the individual having lowest fitness score ie.
            # 0 then we know that we have reached to the targeta
            # and break the loop
        if population[0].fitness >= 0.8:
            found = True
            break
# Otherwise generate new offsprings for new generation
        new_generation = []
# Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])
# From 50% of fittest population, Individuals
        # will mate to produce offspring
        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)
        population = new_generation
        print("Best by now:",generation, population[0].chromosome)
        generation += 1
        
if __name__ == '__main__':
    main()
