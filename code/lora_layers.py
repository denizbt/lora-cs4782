import torch
import torch.nn as nn

class LoRALayer(nn.Module):
  def __init__(self, wo_layer, rank: int, alpha: int) -> None:
    """
    LoRA layer, performs low-rank adaptation of given HuggingFace model
    
    Args:
      wo_layer: the original attention key or query layer to apply low rank adaptation to.
      rank (int): the low-rank value to use for decomposition
      alpha (int): constant, part of scaling factor for BA
    """
    super().__init__()
    
    self.wo_layer = wo_layer

    # freeze weights and bias for this layer
    # LoRA only trains low-rank matrices A, B
    self.wo_layer.weight.requires_grad = False
    
    # section 4.2: paper only adapts attention weights (not biases)
    if hasattr(self.wo_layer, "bias") and self.wo_layer.bias is not None:
      self.wo_layer.bias.requires_grad = False

    # the low-rank value for weight matrix decomposition
    self.r = rank
    
    # constant, part of scaling factor
    self.alpha = alpha
    
    # used to scale BAx 
    self.scale_factor = self.alpha / self.r
    
    # d_in: first dimension of pre-trained weight matrix
    # k_out: second dimension of pre-trained weight matrix Wo
    d_in = self.wo_layer.in_features
    k_out = self.wo_layer.out_features

    # low-rank matrices
    self.B = nn.Parameter(torch.zeros((d_in, self.r)), requires_grad=True) # initialize to zero
    self.A = nn.Parameter(torch.normal(0, 1, size=(self.r, k_out)), requires_grad=True) # initialize from random Gaussian

  def forward(self, x):
    # modified forward pass, eqn (3) from paper
    # h = W_ox + BAx
    Wo_out = self.wo_layer(x)
    lora_out = self.scale_factor * (x @ self.B @ self.A)

    return Wo_out + lora_out

def inject_lora_to_kq_attn(args, model, rank=8, alpha=8):
    """
    Given a model, replaces key and query attention layers with a custom LoRA layer (defined in lora_layers.py)
    and freezes all parameters except for LoRA matrices i.e. injects LoRA

    Args:
      model (nn.Module): the model to inject LoRA matrices to
    """
    # freeze all parameters (including encoder and MLP)!
    # this goes for every model, following paper which only fine-tunes the attention weights
    for param in model.parameters():
        param.requires_grad = False

    if args.model_name == "roberta-base":
      for layer in model.roberta.encoder.layer:
          attn = layer.attention.self
          if hasattr(attn, "query") and hasattr(attn, "key"):
              if isinstance(attn.query, nn.Linear) and isinstance(attn.key, nn.Linear):
                  lora_query = LoRALayer(attn.query, rank=rank, alpha=alpha)
                  attn.query = lora_query
                  
                  lora_key = LoRALayer(attn.key, rank=rank, alpha=alpha)
                  attn.key = lora_key
    elif "microsoft/deberta" in args.model_name:
      for layer in model.deberta.encoder.layer:
          if hasattr(layer, "attention"):
              attn = layer.attention
              
              # apply LoRA to query matriices
              if hasattr(attn, "self") and hasattr(attn.self, "query_proj"):
                  if isinstance(attn.self.query_proj, nn.Linear):
                      lora_query = LoRALayer(attn.self.query_proj, rank=rank, alpha=alpha)
                      attn.self.query_proj = lora_query
              
              # apply LoRA to key matrices
              if hasattr(attn, "self") and hasattr(attn.self, "key_proj"):
                  if isinstance(attn.self.key_proj, nn.Linear):
                      lora_key = LoRALayer(attn.self.key_proj, rank=rank, alpha=alpha)
                      attn.self.key_proj = lora_key
              
              # apply LoRA to value matrices
              # if hasattr(attn, "self") and hasattr(attn.self, "value_proj"):
              #     if isinstance(attn.self.value_proj, nn.Linear):
              #         lora_val = LoRALayer(attn.self.value_proj, rank=rank, alpha=alpha)
              #         attn.self.value_proj = lora_val
    else:
      raise RuntimeError(f"Model type {args.model_name} is not supported by this implementation of LoRA.")