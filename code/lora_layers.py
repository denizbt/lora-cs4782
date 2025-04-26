import torch
import torch.nn as nn

class LoRALayer(nn.Module):
  def __init__(self, wo_layer: nn.Linear, rank: int, alpha: int) -> None:
    """
    LoRA layer, performs low-rank adaptation of given HuggingFace model
    
    Args:
      wo_layer (nn.Linear): the original attention key or query layer to apply low rank adaptation to.
      rank (int): the low-rank value to use for decomposition
      alpha (int): constant, part of scaling factor for BA
    """
    super().__init__()
    
    self.wo_layer = wo_layer
    # freeze weights and bias for this layer
    # LoRA only trains low-rank matrices A, B
    self.wo_layer.weight.requires_grad = False
    
    # section 4.2: paper only adapts attention weights (not biases)
    if hasattr(self.wo_layer, "bias"):
      self.wo_layer.bias.requires_grad = False

    # the low-rank value for weight matrix decomposition
    self.r = rank
    
    # constant, part of scaling factor
    self.alpha = alpha
    
    # used to scale (BAx) 
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

    # x.size() --> [batch_size, seq_length, embedding_dim]
    batch_size, seq_length, embed_dim = x.shape
    x_flatten = x.view(-1, embed_dim)  # [batch_size*seq_length, embedding_dim]
    
    # multiply by LORA matrices
    lora_out = x_flatten @ (self.B @ self.A)  # [batch*seq, embed_dim] @ ([embed_dim, r] @ [r, output_dim])
    
    # recover batch dimension
    lora_out = lora_out.view(batch_size, seq_length, -1)

    return Wo_out + self.scale_factor * lora_out

def inject_lora_to_kq_attn(args, model, rank=8, alpha=8):
    """
    Given a model, replaces key and query attention layers with a custom LoRA layer (defined in lora_layers.py)
    and freezes all parameters except for LoRA matrices i.e. injects LoRA

    Args:
      args (Namespace): command line-arguments to this script
      model (nn.Module): the model to inject LoRA matrices to
    """
    if args.model_name == "roberta-base":
      # freeze all parameters
      for param in model.roberta.parameters():
          param.requires_grad = False

      for layer in model.roberta.encoder.layer:
          attn = layer.attention.self
          if hasattr(attn, "query") and hasattr(attn, "key"):
              if isinstance(attn.query, nn.Linear) and isinstance(attn.key, nn.Linear):
                  lora_query = LoRALayer(attn.query, rank=rank, alpha=alpha)
                  attn.query = lora_query
                  
                  lora_key = LoRALayer(attn.key, rank=rank, alpha=alpha)
                  attn.key = lora_key
    elif args.model_name == "deberta-base":
      raise NotImplementedError()
    else:
      raise RuntimeError("Model type is not supported by this implementation of LoRA.")