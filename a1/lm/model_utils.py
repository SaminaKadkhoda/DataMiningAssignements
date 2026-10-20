import torch
import math
import numpy as np
from lm.model import softmax 
def cross_entropy(logits, targets):
    logits = logits.to(torch.float32)
    m = logits.max(-1, keepdim=True)[0]
    z = logits - m
    lse = z.exp().sum(-1).log()
    tgt = z.gather(-1, targets[..., None])
    return (lse - tgt[..., 0]).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            params = group["params"]
            for param in params:
                if param.grad is None: #we don't manually initialize grad, so skip this iter
                    continue
                g = param.grad
                state = self.state[param] #state of param
                if len(state) == 0: #if no state we initialize state
                    state["step"]=0 #steps start from 0
                    state["m"] = torch.zeros_like(param) #first moment is set to 0, matching params format
                    state["v"] = torch.zeros_like(param) #second moment is set to 0, matching params format

                state["step"]+=1 #we are running optimizer so one step more
                #updating moments and step
                m = state["m"] 
                v = state["v"]
                step = state["step"]

                m = b1*m + (1-b1)*g
                v = b2*v + (1-b2)*(g**2)
                state["m"] = m
                state["v"] = v
                bias_m = m/(1-b1**step)
                bias_v = v/(1-b2**step)
                param_next = param - lr*((bias_m/(bias_v.sqrt()+eps)) + weight_decay*param)
                param.copy_(param_next)

def lr_cosine(step, lr_max, lr_min, warmup_steps, total_steps):
    if step <= warmup_steps and warmup_steps > 0: #linear warmup and skip if no warmup
        return lr_max*step/warmup_steps
    if step >= total_steps: #constant afterwards
        return lr_min
    #decay to min
    angle = math.pi *(step-warmup_steps)/(total_steps-warmup_steps) 
    return lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(angle))

def clip_grad_norm_(params, max_norm, eps=1e-6):
    grads = [param.grad for param in params if param.grad is not None]
    global_norm = 0
    for g in grads:
        global_norm += (g**2).sum()
    global_norm = global_norm.sqrt()
    if global_norm > max_norm:
        scale = max_norm/(global_norm+eps)
        for g in grads:
            g *= scale
    return global_norm

def get_batch(data, batch_size, context_length, device):
    #randomly choosing batch_size starting index in data, from the first token to the last-context_length
    random_idx = np.random.randint(0, len(data)-context_length, batch_size)
    x = []
    y = []
    for i in random_idx:
        x.append(data[i:i+context_length])
        y.append(data[i+1:i+context_length+1]) #for predicting the next token(+1)
    x = torch.tensor(np.array(x), dtype=torch.int64, device=device)
    y = torch.tensor(np.array(y), dtype=torch.int64, device=device)
    return x, y
    
def save_checkpoint(model, optimizer, step, path):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    torch.save(obj, path)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]

def nucleus(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    removed = torch.zeros_like(sorted_probs, dtype=torch.bool) #same format as probs btu set to zero
    total_p = 0.0
    for i, p in enumerate(sorted_probs[0]):
        if total_p >= top_p:
            removed[0,i] = True
        else:
            total_p += p.item()
    sorted_probs[removed] = 0.0
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(1, sorted_indices, sorted_probs)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    return filtered_probs

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_p, eos_id=None, seed=0):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = []
    generator = torch.Generator(device=device).manual_seed(seed)

    for _ in range(max_new_tokens):
        x = ids[:, -model.context_length:]
        logits = model(x)[:,-1,:]
        probs = softmax(logits/temperature, dim=-1)
        probs = nucleus(probs, top_p)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        token_id = next_id.item()
        generated.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break
        ids = torch.cat([ids, next_id], dim=1)
    return generated