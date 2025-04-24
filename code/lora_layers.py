import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class LoRALayer(nn.Module):
  def __init__(self, model_name, d_in, k_out, rank, alpha) -> None:
    """
    LoRA layer, performs low-rank adaptation of given HuggingFace model
    
    Args:
      model_name (str): name of model to load from AutoModel
      d_in (int): first dimension of pre-trained weight matrix Wo
      k_out (int): second dimension of pre-trained weight matrix Wo
      rank (int): the low-rank value to use for decomposition
      alpha (int): constant, part of scaling factor for Wo
    """
    super().__init__()
    
    # pre-trained model
    self.model = AutoModel.from_pretrained(model_name)
    
    # pre-trained weight matrix (using Wq only for now)
    # TODO check if this structure is same for all pre-trained models we are going to use this on
    last_attention = self.model.encoder.layer[len(self.model.encoder.layer) - 1].attention.self
    self.Wq = last_attention.query.weight.detach().numpy()

    # the low-rank value for weight matrix decomposition
    self.r = rank
    
    # alpha, part of scaling factor alpha/r
    self.alpha = alpha
    
    self.scale_factor = self.alpha / self.r

    # low-rank matrices
    self.B = nn.Parameter(torch.zeros((d_in, self.r))) # init to zero
    self.A = nn.Parameter(torch.normal(0, 1, size=(self.r, k_out))) # init from random Gaussian

  def forward(self, x):
    # TODO check the dimensions of x to make sure it works out
    h = self.Wq @ x + self.scale_factor * ((self.B @ self.A) @ x)
    return h