import torch.nn.functional as F
import torch.nn as nn
from utils.metrics import dice_coef
import numpy as np
import torch


def sample_eval(x, y, model, Nsamples = 20, logits=True, train=False):

    out = model.sample_predict(x, Nsamples)

    if logits:
        mean_out = out.mean(dim=0, keepdim=False)
        loss = F.cross_entropy(mean_out, y[:, 0, :, :].long(), reduction='sum')
        probs = F.softmax(mean_out, dim=1).data.cpu()

    else:
        mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
        probs = mean_out.data.cpu()

        log_mean_probs_out = torch.log(mean_out)
        loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

    predict = probs[:, 1, :, :].unsqueeze(1)

    dice_score = dice_coef(predict, y)

    return loss.data, dice_score, probs, mean_out


def all_sample_eval(self, x, Nsamples=20):

    out = self.model.sample_predict(x, Nsamples)

    prob_out = F.softmax(out, dim=2)
    prob_out = prob_out.data

    return prob_out


def get_weight_samples(self):
    weight_vec = []

    state_dict = self.model.state_dict()

    for key in state_dict.keys():

        if 'weight' in key:
            weight_mtx = state_dict[key].cpu().data
            for weight in weight_mtx.view(-1):
                weight_vec.append(weight)

    return np.array(weight_vec)