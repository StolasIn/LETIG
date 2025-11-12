from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.sms import SMSEMOA

from pymoo.core.result import Result
from pymoo.core.problem import Problem
import torch
import numpy as np

from EvaluationModel import EvaluationModel
from Generator import Generator
from PIL import Image


class Problems(Problem):
    def __init__(
        self, 
        evaluation_function, 
        variable_size,
        n_objectives,
        lower_bound = -2,
        upper_bound = 2,
        device = 'cpu'
    ):
        super().__init__(
            n_var = variable_size,
            n_obj = n_objectives,
            xl = lower_bound,
            xu = upper_bound
        )

        self.device = device
        self.evaluation_function = evaluation_function

    def fitness(self, scores):
        if self.n_obj == 1:
            return [sum(scores[i]) for i in range(len(scores))]
        else:
            return np.array(scores)

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.evaluation_function(ws = torch.from_numpy(x).to(self.device))
        out['F'] = self.fitness(scores)

class Optimizer:
    def __init__(
        self,
        generator_name,
        use_txt_feature = False,
        device = 'cpu'
    ):
        torch.autograd.set_grad_enabled(False)
        self.device = device
        
        # Generator
        self.generator_path = f'checkpoints/{generator_name}'
        self.use_fts = use_txt_feature
        
        # classes
        self.generator = Generator(self.generator_path, self.use_fts, self.device)
        self.evaluator = EvaluationModel()

    def setup_evaluator(
        self, 
        evaluator_info,
        prompt_text: str = None,
        prompt_image: Image.Image = None,
        n_dataset_samples = 20,
        dataset_threshold = 0.25
    ):
        if prompt_text is None and prompt_image is None:
            raise ValueError("Either prompt_text or prompt_image must be provided.")

        self.evaluator.setup(prompt_text, prompt_image, evaluator_info, n_dataset_samples, dataset_threshold, self.device)

    def setup_optimizer(
        self, 
        algorithm_name,
        lower_bound,
        upper_bound,
        population_size,
        n_objectives,
        n_evaluations
    ):
        self.algorithm = self.create_algorithm(
            algorithm_name = algorithm_name,
            lower_bound = lower_bound,
            upper_bound = upper_bound,
            population_size = population_size
        )

        self.problem = Problems(
            self.get_scores, 
            variable_size = self.generator.get_variable_length(),
            n_objectives = n_objectives,
            lower_bound = lower_bound,
            upper_bound = upper_bound,
            device = self.device
        )

        self.n_evaluations = n_evaluations

    def get_scores(self, ws):
        images = [self.generator.get_img_from_w(w = w) for w in ws]
        scores = self.evaluator.calculate_similaritys(images, batch_size = len(images))

        return scores
    
    def create_algorithm(
        self, 
        algorithm_name,
        lower_bound,
        upper_bound,
        population_size
    ):
    
        algorithm = None

        if algorithm_name == 'CMAES':
            algorithm = CMAES(
                x0 = np.random.uniform(lower_bound, upper_bound, size=(self.generator.get_variable_length())),
                sigma = 0.5,
                pop_size = population_size
            )
        elif algorithm_name == "G3PCX":
            algorithm = G3PCX(
                pop_size = population_size
            )
        elif algorithm_name == 'NSGA2':
            algorithm = NSGA2(
                pop_size = population_size
            )
        elif algorithm_name == "AGEMOEA2":
            algorithm = AGEMOEA2(
                pop_size = population_size
            )
        elif algorithm_name == "SMSEMOA":
            algorithm = SMSEMOA(
                pop_size = population_size
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm_name} not implemented.")
        return algorithm

    def solve(self, verbose = False, seed = -1):
        res = minimize(
            self.problem, 
            self.algorithm, 
            termination = ('n_evals', self.n_evaluations), 
            copy_algorithm = False,
            seed = seed if seed != -1 else None,
            verbose = verbose
        )

        print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
        return res.X, res.F

    def get_fes(self, verbose = False, seed = -1):
        x, f = self.solve(verbose, seed)
        w = torch.from_numpy(x).to(self.device)
        image = self.generator.get_img_from_w(w = w)
        return image, x, f