import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist

# bs can be batch size or buffer size
def init_random(bs, img_size):
    n_ch, dim1, dim2 = img_size
    return torch.FloatTensor(bs, n_ch, dim1, dim2).uniform_(-1, 1)

def getbuffer(args, img_size, device):
    replay_buffer = init_random(args.buffer_size, img_size)
    return replay_buffer

def sample_p_0(args, replay_buffer, img_size, bs, device):
    if len(replay_buffer) == 0:
        return init_random(bs, img_size), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, img_size)
    choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds

def sample_q(args, model, img_size, replay_buffer, device, n_steps):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    model.eval()
    # get batch size
    bs = args.batch_size
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(args, replay_buffer, img_size, bs, device)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        out_k = model(x_k)
        logsumexp_k = out_k.logsumexp(1)
        f_prime = torch.autograd.grad(logsumexp_k.sum((1,2)).sum(0), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
    model.train()
    final_samples = x_k.detach()
    # update replay buffer?
    if args.buffer_type == 'persistent':
        replay_buffer[buffer_inds] = final_samples.cpu()
    return replay_buffer, final_samples