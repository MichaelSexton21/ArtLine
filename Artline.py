import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
import torchvision
import sys
import warnings


# %matplotlib inline

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    path = Path(".")


    imgPath = sys.argv[1]
    img = PIL.Image.open(imgPath).convert("RGB")

    modelHeight = 'ArtLine_650.pkl'
    if sys.argv[2] == "920":
        modelHeight = "ArtLine_920.pkl"
        
    learn=load_learner(path, modelHeight) 
    
    img_t = T.ToTensor()(img)
    img_fast = Image(img_t)
    show_image(img_fast, figsize=(8,8), interpolation='nearest');

    p,img_hr,b = learn.predict(img_fast)
    show_image(img_hr, figsize=(9,9), interpolation='nearest')

    outputFileName = (imgPath.split('/')[-1].split('.')[0]).capitalize()

    torchvision.utils.save_image(img_hr, "../results/line" + outputFileName + ".png")

