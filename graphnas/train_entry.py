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
from graphnas.search_space import MacroSearchSpace

ssl._create_default_https_context = ssl._create_unverified_context


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


def main(args):  # pylint:disable=redefined-outer-name

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

    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'derive':
        trnr.derive()
    elif args.mode == 'genetic_algorithm':
        trnr.genetic_get_reward()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")


def fitness_func(solution, solution_idx):
    args = build_args()
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    utils.makedirs(args.dataset)
    global trnr

    search_space = MacroSearchSpace().search_space
    gnn = []
    gnn.append(search_space['attention_type'][int(solution[0])])
    gnn.append(search_space['aggregator_type'][int(solution[1])])
    gnn.append(search_space['activate_function'][int(solution[2])])
    gnn.append(int(solution[3]))
    gnn.append(int(solution[4]))
    gnn.append(search_space['attention_type'][int(solution[5])])
    gnn.append(search_space['aggregator_type'][int(solution[6])])
    gnn.append(search_space['activate_function'][int(solution[7])])
    gnn.append(int(solution[8]))
    gnn.append(6)
    fitness = trnr.genetic_get_reward(gnn)[1]
    return fitness


def callback_gen(ga_instance):
    with open(args.dataset + "_" + args.search_mode + args.submanager_log_file, "a") as file:
        # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
        file.write("Generation : ")
        file.write(str(ga_instance.generations_completed))
        file.write("\n")
        file.write("Fitness of the best solution :")
        file.write(str(ga_instance.best_solution()[1]))
        file.write("\n")
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


for i in range(1):
    total_time = time.time()
    args = build_args()
    num_generations = 200
    num_parents_mating = 4
    sol_per_pop = 8
    parent_selection_type = "sss"
    keep_parents = 1
    with open(args.dataset + "_" + args.search_mode + args.submanager_log_file, "a") as file:
        # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
        file.write("num_generations = ")
        file.write(str(num_generations))
        file.write("num_parents_mating = ")
        file.write(str(num_parents_mating))
        file.write("sol_per_pop = ")
        file.write(str(sol_per_pop))
        file.write("parent_selection_type = ")
        file.write(str(parent_selection_type))
        file.write("keep_parents = ")
        file.write(str(keep_parents))
        file.write("\n")

    crossover_type = "single_point"
    fitness_function = fitness_func
    attention_type_gene_space = [0, 1, 2, 3, 4, 5, 6]
    aggregator_type_gene_space = [0, 1, 2, 3]
    activate_function_gene_space = [0, 1, 2, 3, 4, 5, 6, 7]
    number_of_heads_gene_space = [1, 2, 4, 6, 8, 16]
    hidden_units_gene_space = [4, 8, 16, 32, 64, 128, 256]
    gene_space = [attention_type_gene_space, aggregator_type_gene_space, activate_function_gene_space,
                  number_of_heads_gene_space, hidden_units_gene_space, attention_type_gene_space,
                  aggregator_type_gene_space, activate_function_gene_space, number_of_heads_gene_space]
    num_genes = len(gene_space)
    mutation_type = "random"
    mutation_percent_genes = 10

    trnr = trainer.Trainer(args)
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           gene_type=int,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           on_generation=callback_gen,
                           mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()
    ga_instance.plot_result()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    with open(args.dataset + "_" + args.search_mode + args.submanager_log_file, "a") as file:
        # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
        file.write("Parameters of the best solution : ")
        file.write(str(solution))
        file.write("Fitness value of the best solution = ")
        file.write(str(solution_fitness))
        file.write("Index of the best solution : ")
        file.write(str(solution_idx))
        file.write("The total training time: ")
        file.write(str(total_time))
        file.write("\n")
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
