import torch
from torch import nn
import math
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma = (2 / (in_features + out_features)) ** 0.5 
        nn.init.trunc_normal_(self.w, mean=0.0, std=self.sigma, a=-3*self.sigma, b=3*self.sigma)
    
    def forward(self, x):
        return torch.einsum("... i, o i -> ... o", x, self.w)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.w, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, ids):
        return self.w[ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))        
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        y = (x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps))*self.g
        return y.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

        for w in (self.w1, self.w2, self.w3):
            sigma = (2 / (w.shape[0] + w.shape[1])) ** 0.5
            nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x):
        w1x = torch.einsum("... i, o i -> ... o", x, self.w1)
        w3x = torch.einsum("... i, o i -> ... o", x, self.w3)

        w1x_silu = torch.nn.functional.silu(w1x)
        w1x_silu_w3x = w1x_silu * w3x
        return torch.einsum("... i, o i -> ... o", w1x_silu_w3x, self.w2)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len):
        super().__init__()
        i = torch.arange(0, d_k, 2) 
        pos = torch.arange(0,max_seq_len)
        theta_i = theta**(-i/d_k)
        angles = torch.einsum('i,j->ij', pos, theta_i)
        self.register_buffer("cos_table", torch.cos(angles), persistent=False)
        self.register_buffer("sin_table", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        if token_positions.ndim == 2:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)

def softmax(x, dim):
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x-max_x)  
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    qk = torch.einsum('... i d, ... j d -> ... i j', q, k)
    sf_x = qk/math.sqrt(d_k)
    if mask is not None:
        sf_x = sf_x.masked_fill(~mask, float("-inf")) #replace values if mask false
    softmax_res = softmax(sf_x.to(torch.float32), -1)
    return torch.einsum('... i j, ... j d-> ... i d', softmax_res, v)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, rope=None):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads

        if d_model%num_kv_heads != 0 or d_model%num_heads:
            raise ValueError("d_model must be divisible by num_heads/num_kv_heads!")

        if num_heads%num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads!")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.rope = rope
        self.d_h = d_model//num_heads

        self.w_q = Linear(d_model, num_heads * self.d_h)
        self.w_k = Linear(d_model, num_kv_heads * self.d_h)
        self.w_v = Linear(d_model, num_kv_heads * self.d_h)
        self.w_o = Linear(d_model, d_model)
    
    def forward(self, x, token_positions):
        leading_shape = x.shape[:-2]
        s = x.shape[-2]

        q = self.w_q(x).reshape(*leading_shape, s, self.num_heads, self.d_h)
        k = self.w_k(x).reshape(*leading_shape, s, self.num_kv_heads, self.d_h)
        v = self.w_v(x).reshape(*leading_shape, s, self.num_kv_heads, self.d_h)

        # Move heads before sequence dimension
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if self.num_kv_heads < self.num_heads: #if kv heads are less than query heads, then reuse them
            n = self.num_heads//self.num_kv_heads
            k = k.repeat_interleave(n, dim=-3)
            v = v.repeat_interleave(n, dim=-3)

        mask = torch.tril(torch.ones(s, s, dtype=bool, device=x.device))
        #o = scaled_dot_product_attention(q, k, v, mask).transpose(-3,-2).contiguous().reshape(*leading_shape, s, self.num_heads * self.d_h)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0).transpose(-3,-2).contiguous().reshape(*leading_shape, s, self.num_heads * self.d_h) #for efficinecy table
        

        return self.w_o(o)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, rope=None, eps=1e-5):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model, eps)
        self.attn = MultiheadSelfAttention(d_model, num_heads, num_kv_heads=num_kv_heads, rope=rope)
        self.rmsnorm2 = RMSNorm(d_model, eps)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions):
        h = x + self.attn(self.rmsnorm1(x), token_positions)
        y = h + self.swiglu(self.rmsnorm2(h))
        return y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, num_kv_heads=None, tied=False, rope=None):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.context_length = context_length
        self.embeddings = Embedding(vocab_size, d_model)
        self.rmsnorm_final = RMSNorm(d_model)
        self.output_head = Linear(d_model, vocab_size)
        if tied:
            self.output_head.w = self.embeddings.w
        if rope is None:
            rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_model//num_heads, max_seq_len=context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_kv_heads=num_kv_heads, rope=rope)for _ in range(num_layers)])

    
    def forward(self, ids):
        token_positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embeddings(ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.rmsnorm_final(x)
        logits = self.output_head(x).to(torch.float32)
        return logits

def count_params(model):
    p_count = 0
    for param in model.parameters():
        p_count += torch.numel(param)
    return p_count

def count_flops(config, seq_len):
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    d_ff = config["d_ff"]
    num_heads = config["num_heads"]
    num_kv_heads = config.get("num_kv_heads", num_heads)

    d_head = d_model // num_heads

    #2*width*Din*Dout
    proj_flops = 8*num_layers*(d_model**2)
    attn_flops = 4*num_layers*seq_len*d_model
    mlp_flops = 6*num_layers*d_model*d_ff
    head_flops = 2*d_model*vocab_size

    forward_flops = proj_flops + attn_flops + mlp_flops + head_flops
    training_flops = 3*forward_flops

    return forward_flops, training_flops