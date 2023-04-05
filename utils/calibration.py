import torch
from torch import nn, optim
from torch.nn import functional as F

class ece_function(nn.Module):
    def __init__(self, n_bins, device):
        super(ece_function, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        
        self.n_bins = n_bins
        self.count = torch.zeros(n_bins).to(device)
        self.acc = torch.zeros(n_bins).to(device)
        self.conf = torch.zeros(n_bins).to(device)

    def forward(self, probs, labels):
        confidences, predictions = torch.max(probs, 1)
        #confidences = confidences * predictions
        confidences = confidences[0,:,:]
        confidences = confidences.view(-1) # h x w
        predictions = predictions[0,:,:]
        predictions = predictions.view(-1) # h x w
        labels = labels[0,0,:,:]
        labels = labels.view(-1) # h x w

        predictions = predictions.cuda()
        accuracies = predictions.eq(labels)
        
        for i in range(self.n_bins):
            bin_lower = self.bin_lowers[i]
            bin_upper = self.bin_uppers[i]
            # Calculated |confidence - accuracy| in each bin

            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

            self.count[i] = in_bin.float().sum().item()
            self.conf[i] = confidences[in_bin].sum().item()
            self.acc[i] = accuracies.float()[in_bin].sum().item()

        return self.count, self.conf, self.acc
    
class sce_function(nn.Module):
    def __init__(self, n_bins, device):
        super(sce_function, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        
        self.n_bins = n_bins
        self.count0 = torch.zeros(n_bins).to(device)
        self.acc0 = torch.zeros(n_bins).to(device)
        self.conf0 = torch.zeros(n_bins).to(device)
        self.count1 = torch.zeros(n_bins).to(device)
        self.acc1 = torch.zeros(n_bins).to(device)
        self.conf1 = torch.zeros(n_bins).to(device)

    def forward(self, probs, labels):
        _,_,h,w = probs.shape
        confidences0 = probs[0,0,:,:]
        confidences0 = confidences0.reshape(h*w) # h x w
        confidences1 = probs[0,1,:,:]
        confidences1 = confidences1.reshape(h*w)
        labels = labels[0,0,:,:]
        labels = labels.reshape(h*w) # h x w

        accuracies0 = (labels == 0) * 1.
        accuracies1 = (labels == 1) * 1.
        
        for i in range(self.n_bins):
            bin_lower = self.bin_lowers[i]
            bin_upper = self.bin_uppers[i]
            # Calculated |confidence - accuracy| in each bin

            in_bin0 = confidences0.gt(bin_lower.item()) * confidences0.le(bin_upper.item())

            self.count0[i] = in_bin0.float().sum().item()
            self.conf0[i] = confidences0[in_bin0].sum().item()
            self.acc0[i] = accuracies0.float()[in_bin0].sum().item()

            in_bin1 = confidences1.gt(bin_lower.item()) * confidences1.le(bin_upper.item())

            self.count1[i] = in_bin1.float().sum().item()
            self.conf1[i] = confidences1[in_bin1].sum().item()
            self.acc1[i] = accuracies1.float()[in_bin1].sum().item()

        return self.count0, self.conf0, self.acc0, self.count1, self.conf1, self.acc1