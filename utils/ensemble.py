import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.metrics import dice_coef


def get_weight_samples(checkpoint):
    weight_vec = []

    state_dict = checkpoint
    for i in range(5):
        weight_mtx = checkpoint[i].data
        for weight in weight_mtx.view(-1):
            weight_vec.append(weight)

    return np.array(weight_vec)


def ensemble_result(x, y, model, weight_set_samples):
    Nsamples = len(weight_set_samples)

    out = x.data.new(Nsamples, x.shape[0], 2, 128, 128)
    true_probs = x.data.new(Nsamples, x.shape[0], 1, 128, 128)
    false_probs = x.data.new(Nsamples, x.shape[0], 1, 128, 128)

    # iterate over all saved weight configuration samples
    for idx, weight_dict in enumerate(weight_set_samples):
        if idx == Nsamples:
            break
        model.load_state_dict(weight_dict)
        logit = model(x)
        out[idx] = logit

        prob = F.softmax(logit, dim=1).data.cpu()
        true_prob = prob[:,1,:,:]
        false_prob = prob[:,0,:,:]
        true_probs[idx] = true_prob
        false_probs[idx] = false_prob

    mean_out = out.mean(dim=0, keepdim=False)
    true_mean_prob = true_probs.mean(dim=0, keepdim=False)
    false_mean_prob = false_probs.mean(dim=0, keepdim=False)

    probs = torch.cat((true_mean_prob, false_mean_prob), 1)

    dice_score = dice_coef(true_mean_prob, y)
    return dice_score, probs, mean_out


def all_sample_eval(self, x, y, weight_set_samples):
    Nsamples = len(weight_set_samples)

    out = x.data.new(Nsamples, x.shape[0], self.classes)

    # iterate over all saved weight configuration samples
    for idx, weight_dict in enumerate(weight_set_samples):
        if idx == Nsamples:
            break
        self.model.load_state_dict(weight_dict)
        out[idx] = self.model(x)

    prob_out = F.softmax(out, dim=2)
    prob_out = prob_out.data

    return prob_out