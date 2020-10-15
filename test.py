
import random
from eval_scripts.semi.eval_designed_gnn import eval
# Number of individuals in each generation
POPULATION_SIZE = 100



search_space = [
["gat", "gcn", "cos", "const", "gat_sym", "linear", "generalized_linear"],
["sum", "mean", "max", "mlp"],
["sigmoid", "tanh", "relu", "linear", "softplus", "leaky_relu", "relu6", "elu"],
[1, 2, 4, 6, 8, 16],
[4, 8, 16, 32, 64, 128,256],
]

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
                num_within_layer = random.randrange(1,3)
                layer = generate_layer(num_within_layer)
                self.chromosome.append(layer)
        else:
            self.chromosome = chromosome
            self.num_layer = len(chromosome)
        print("individual:",self.chromosome)
        self.fitness = eval(self.chromosome)



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
    def cal_fitness(self):
        '''
        Calculate fittness score, it is the number of
        characters in string which differ from target
        string.
        '''
        fitness = eval_smi(self.chromosome)
        return fitness
# Driver code
def main():

# current generation
    generation = 1
    print("-"*40,"generation:",generation,"-"*40)
    found = False
    population = []
# create initial population
    for _ in range(POPULATION_SIZE):
        population.append(Individual())
    while not found:
# sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness)
    # if the individual having lowest fitness score ie.
            # 0 then we know that we have reached to the target
            # and break the loop
        print("Best by now", population[0].chromosome,population[0].fitness)
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
        
        generation += 1
        
if __name__ == '__main__':
    main()
