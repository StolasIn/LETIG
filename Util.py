from PIL import Image
import torch
from Generator import Generator
from EvaluationModel import EvaluationModel
import Optimizer

class Util:
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

    def setup(
        self, 
        evaluator_info,
        prompt_text: str = None,
        prompt_image: Image.Image = None
    ):
        if prompt_text is None and prompt_image is None:
            raise ValueError("Either prompt_text or prompt_image must be provided.")
        
        self.evaluator.setup(prompt_text, prompt_image, evaluator_info, self.device)

    def get_scores(self, ws):
        text_semantic_scores = []
        image_semantic_scores = []
        text_realistic_scores = []
        image_realistic_scores = []
        scores = []
        for w in ws:
            img = self.generator.get_img_from_w(w = w)
            scores = self.evaluator.calculate_similaritys(img)
            text_semantic_score, image_semantic_score, text_realistic_score, image_realistic_score = scores[0], scores[1], scores[2], scores[3]
            
            # -1 for maximization
            text_semantic_scores.append(-text_semantic_score)
            image_semantic_scores.append(-image_semantic_score)
            text_realistic_scores.append(-text_realistic_score)
            image_realistic_scores.append(-image_realistic_score)

        return text_semantic_scores, image_semantic_scores, text_realistic_scores, image_realistic_scores
        
    def get_fes(self):
        x, f = Optimizer.solve(self, self.config)
        w = torch.from_numpy(x).to(self.device)
        image = self.generator.get_img_from_w(w = w)
        return image, x, f