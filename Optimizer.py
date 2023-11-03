from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

from pymoo.core.result import Result
from pymoo.core.problem import Problem
import torch
import numpy as np


class CMAES_MultiTask(CMAES):
    
    """
        run one step of optimization, rather than whole process
        override CMAES class function
    """
    
    def run(self):
        res = Result()
        self.iterator = self._run(self.problem)
        next(self.iterator)
        res = self.result()
        return res
    
    def _run(self, problem):
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")
        
        while self.has_next():
            self.next()
            yield
        yield True

class Tasks:
    
    """
        handle problems and algorithms for tasks
        
        paramters
        =========
        
        problems : a batch of Problems class (Problems, [n])
        algorithms : a batch of CMAES_MultiTask class (CMAES_MultiTask, [n])
        iterators : a batch of iterators from prepare_iterators (algorithms iterator, [n])
    """
    
    def __init__(
        self,
        problems,
        algorithms,
        iterators
    ):
        self.Problems = problems
        self.Algorithms = algorithms
        self.Iterators = iterators

        self.task_len = len(problems)
        self.offspring_size = len(self.Algorithms[0].ask())
        self.terminals = np.zeros(self.task_len, dtype = bool)
        
    # move one step of optimization process
    def move(self):
        for i in range(self.task_len):
            if self.terminals[i] == True:
                continue
            
            iterator = self.Iterators[i]
            terminal = next(iterator)
            if terminal == True:
                self.terminals[i] = True
        
        return all(self.terminals) # if all tasks is complete
        
    def get_offspring(self):
        offspring_list = []
        for i in range(self.task_len):
            offspring = self.get_offspring_of_task(i)
            offspring_list.append(offspring)
            
        return offspring_list

    def get_offspring_of_task(self, task_index):
        offspring = self.Algorithms[task_index].ask()
        return offspring
    
    # evaluate new offspring
    def evaluation(self, offspring):
        for i in range(self.task_len):
            self.Algorithms[i].evaluator.eval(self.Problems[i], offspring[i])
    
    def set_offspring(self, offspring):
        for i in range(self.task_len):
            self.Algorithms[i].tell(infills = offspring[i])

    def final_result(self):
        results_x = []
        results_f = []
        for algorithm in self.Algorithms:
            res = algorithm.result()
            results_x.append(res.X)
            results_f.append(res.f)
        
        return results_x, results_f

    def get_runs(self):
        xs = []
        fs = []
        for problem in self.Problems:
            x, f = problem.get()
            xs.append(x)
            fs.append(f)
        return xs, fs
        
class Fitness:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.semantic = config['OPTIM'].getfloat('semantic_ratio')
        self.realistic = config['OPTIM'].getfloat('realistic_ratio')

    def fitness(self, scores):
        f = [self.semantic * scores[0][i] + self.realistic * scores[1][i] for i in range(len(scores[0]))]
        f = np.column_stack([f])
        return f

class Record:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.semantic = config['OPTIM'].getfloat('semantic_ratio')
        self.realistic = config['OPTIM'].getfloat('realistic_ratio')
        self.best_x = None
        self.best_f = 0
    
    def record(self, population, scores):
        if len(scores[0]) == 1:
            return
        
        
        scores = [[scores[0][i], scores[1][i]] for i in range(len(scores[0]))]
        self.best_so_far(population, scores)
    
    def get(self):
        return self.best_x, self.best_f*(-1)
    
    def best_so_far(self, population, scores):
        scores = [score[0] + score[1] for score in scores]
        min_index = np.argmin(scores)
        if scores[min_index] < self.best_f:
            self.best_f = scores[min_index]
            self.best_x = population[min_index]
        

class Problems(Problem):
    def __init__(self, util, txt, config):
        super().__init__(
            n_var = config['OPTIM'].getint('n_variables'),
            n_obj = config['OPTIM'].getint('n_objectives'),
            xl = config['OPTIM'].getint('lower_bound'),
            xu = config['OPTIM'].getint('upper_bound')
        )

        self.device = config['BASE']['device']
        self.util = util
        self.txt = txt
        self.F = Fitness(config)
        self.R = Record(config)

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.util.get_scores(txt = self.txt, ws = torch.from_numpy(x).to(self.device))
        self.R.record(x, scores)
        out['F'] = self.F.fitness(scores)
        
    def get(self):
        return self.R.get()

def prepare_problems(util, txts, config):
    problems = []
    for i in range(len(txts)):
        p = Problems(util, txts[i], config)
        problems.append(p)
    return problems

def prepare_algorithms(problems, config):
    xl = config['OPTIM'].getint('lower_bound')
    xu = config['OPTIM'].getint('upper_bound')
    n_variables = config['OPTIM'].getint('n_variables')
 
    algorithms = []
    for i in range(len(problems)):
        algorithm = CMAES_MultiTask(x0=np.random.uniform(xl, xu, size=(n_variables)))
        algorithms.append(algorithm)
    return algorithms

def prepare_iterators(problems, algorithms, config):
    if config['BASE'].getint('seed') == -1:
        seed = None
    else:
        seed = config['BASE'].getint('seed')
  
    iterators = []
    for problem, algorithm in zip(problems, algorithms):
        
        minimize(
            problem, 
            algorithm, 
            termination = ('n_evals', config['OPTIM'].getint('evaluation')), 
            copy_algorithm = False,
            seed = seed,
            verbose=config['BASE'].getboolean('verbose')
        )
        
        iterator = algorithm.iterator
        iterators.append(iterator)
    return iterators

def solve_multitask(util, txts, config):
    
    # build tasks
    problems = prepare_problems(util, txts, config)
    algorithms = prepare_algorithms(problems, config)
    iterators = prepare_iterators(problems, algorithms, config)
    tasks = Tasks(problems, algorithms, iterators)
    
    """
        1. get offspring
        2. evaluaition
        3. set new population
    """
    
    generation = 0
    while len(iterators) > 0:
        terminal = tasks.move()
        if terminal == True:
            break
        
        offspring = tasks.get_offspring()
        tasks.evaluation(offspring)
        tasks.set_offspring(offspring)
        generation+=1
    return tasks.get_runs()