import dnnlib
import legacy
import torch
from PIL import Image
import numpy as np

class Generator:
    
    """
        a StyleGAN based generator (have mapping network)
        
        parameters
        ==========
        
        model_name : path to generator (str)
        use_fts : whether use CLIP text feature to generate image or not (conditional image generation)
        device : gpu or cpu device (str, [cuda:0, cuda:1, cpu])
    """
    
    def __init__(
        self, 
        model_name, 
        use_fts = False, 
        device = 'cuda:0'
    ):
        self.device = device
        self.mapping, self.synthesis = self.load_model(model_name)
        
        self.force_32 = False
        self.use_fts = use_fts
        self.num_ws = self.synthesis.num_ws
        self.z_dim = self.mapping.z_dim
        self.w_avg, self.w_std = self.get_statistic()
    
    def load_model(self, path):
        with dnnlib.util.open_url(path) as f:
            network = legacy.load_network_pkl(f)
            G_ema = network['G_ema'].to(self.device)
            mapping = G_ema.mapping
            synthesis = G_ema.synthesis
            return mapping, synthesis
    
    # get approximate mapping network std and mean
    def get_statistic(self,):
        zs = torch.randn([10000, self.z_dim], device=self.device)
        w_std = self.mapping(zs, None).std(0)
        return self.mapping.w_avg, w_std
    
    def get_w(self,z):
        return self.mapping(z, c = torch.randn((1)).to(self.device))[0][0]

    # generate image tensor form intermediate latent code w 
    def generate_from_w(self, w, c, fts = None, noise_mode='const', return_styles=False):
        w = w * self.w_std + self.w_avg
        w = w.unsqueeze(0)
        if self.use_fts == True:
            img = self.synthesis(ws = w, fts = fts, noise_mode=noise_mode, return_styles=False, force_fp32=self.force_32)
        else:
            img = self.synthesis(ws = w, noise_mode=noise_mode, force_fp32=self.force_32)
        return img

    # generate image tensor form noise (gaussian noise)
    def generate_from_noise(self, z, fts, c = None, noise_mode='const', return_styles=False):
        ws = self.mapping(z, c)
        if self.use_fts == True:
            img = self.synthesis(ws = ws, fts = fts, noise_mode=noise_mode, return_styles=False, force_fp32=self.force_32)
        else:
            img = self.synthesis(ws = ws, noise_mode=noise_mode, force_fp32=self.force_32)
        return img
    
    def get_img_from_tensor(self, tensor):

        """
            convert image tensor to PIL image
            [-1, 1] -> [0, 2] -> [0, 255]
            
            *permute : CWH (image tensor) -> HWC (PIL image)
        """

        img = torch.clamp((tensor + 1.) * 127.5, 0., 255.)
        img = img.permute(1,2,0)
        return Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))
    
    def get_img_from_w(self, w, txt_fts = None):
        ws = w.view(1,-1).to(self.device)
        c = torch.randn((1, 1)).to(self.device)

        if self.use_fts == True:
            txt_fts = txt_fts.view(1,-1).to(self.device)
            img = self.generate_from_w(w=ws, c=c, fts=txt_fts)
        else:
            img = self.generate_from_w(w=ws, c=c)

        img = self.get_img_from_tensor(img[0])
        return img
    
    def get_img_from_noise(self, z, txt_fts = None):
        z = z.view(1,-1).to(self.device)
        c = torch.randn((1, 1)).to(self.device)

        if self.use_fts == True:
            txt_fts = txt_fts.view(1,-1).to(self.device)
            img = self.generate_from_noise(z=z, c=c, fts=txt_fts)
        else:
            img = self.generate_from_noise(z=z, c=c)

        img = self.get_img_from_tensor(img[0])
        return img
    
    # generate random image
    def gen_images(self):
        z = torch.randn((1, 512)).to(self.device)
        img = self.get_img_from_noise(z = z)
        return img