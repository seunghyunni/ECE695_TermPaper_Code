# Import Modules
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.modified_unet import Modified2DUNet
from model.wideresnet import Wide_ResNet
from model.network import R2U_Net, U_Net, ResUnet
from utils.load_data_aug import HeartMRIDataset
import numpy as np
from sklearn.model_selection import train_test_split
from utils.losses import dice_loss
from utils.metrics import dice_coef
from JEM_functions import *
import os
import datetime
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
import easydict
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch 


# Configuration
args = \
easydict.EasyDict({"crop_size": 128,
                #    "resize": 64,
                   "batch_size": 96,
                   "lr": 1e-4,
                   "warmup_iters": 100,
                   "epoch": 1000,
                   "decay_epochs": [100, 200],
                   "decay_rate": 0.8,
                   "loss": 'ce',

                   # SGLD hyperparameters
                   "buffer_type": 'persistent',
                   "buffer_size": 10000,
                   "reinit_freq": .05,
                   "n_steps": 80,
                   "sgld_lr": 0.1,
                   "sgld_std": 0.015,
                   "sigma": 0.15, 
                   
                   # Adjust ratio between losses
                   "seg_weight": 0, 
                   "energy_weight": 1,
                   
                   "sample_freq": 18,
                    "sampledir": './train_results/gen_samples',
                    "ckptdir": './train_results/ckpts',
                    "resumepath": '',
                    "filename": '',
                   })
                   

sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
plot = lambda p, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    # Load Data
    images = glob("./data_preprocessed/preprocessed_images/*.npy")
    labels = glob("./data_preprocessed/preprocessed_labels/*.npy")

    images.sort()
    labels.sort()

    train_images = images[:1200]
    train_labels = labels[:1200]

    valid_images = images[1200:]
    valid_labels = labels[1200:]


    print(len(train_images), len(train_labels), train_images[0], train_labels[0]) 
    print(len(valid_images), len(valid_labels), valid_images[0], valid_labels[0])

    # Dataloader
    # train loader
    img_trans_train = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    # val loader
    img_trans_val = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # mask
    tar_trans = transforms.Compose([
                                    transforms.ToTensor()
                                    ])

    train_data = HeartMRIDataset(args, train_images, train_labels, img_trans_train, img_trans_val, tar_trans, True, -1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, drop_last=True)
                            
    valid_data = HeartMRIDataset(args, valid_images, valid_labels, img_trans_train, img_trans_val, tar_trans, False, -1) 
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Execute only to check the train and test batches.
    ############################################################################################################
    # Train batch
    # fig, ax = plt.subplots(8,8,figsize = (40,40))

    # for i in range(8):
    #     for j in range(8):
    #         data = train_data[j+i*8]
    #         img = data['image'][0]
    #         lab = data['label'][0]
    #         ax[i][j].imshow(img, cmap = 'gray')
    #         ax[i][j].contour(lab, cmap = 'Reds', linewidths = 0.2)
    #         ax[i][j].set_title(data['patient_id']+"_"+data['file_id'])
    #         ax[i][j].axis('off')
    # plt.axis('off')
    # plt.show()
    # fig.savefig("./train_sample.png")
    # plt.close()

    # # Test batch
    # fig, ax = plt.subplots(8,8,figsize = (40,40))

    # for i in range(8):
    #     for j in range(8):
    #         data = valid_data[j+i*8]
    #         img = data['image'][0]
    #         lab = data['label'][0]
    #         ax[i][j].imshow(img, cmap = 'gray')
    #         ax[i][j].contour(lab, cmap = 'Reds', linewidths = 0.2)
    #         ax[i][j].set_title(data['patient_id']+"_"+data['file_id'])
    #         ax[i][j].axis('off')
    # plt.axis('off')
    # plt.show()
    # fig.savefig("./test_sample.png")
    # plt.close()
    ############################################################################################################

    img_size = train_data[0]['image'].size()
    lab_size = train_data[0]['label'].size()
    print("image size: {}".format(img_size))
    print("label size: {}".format(lab_size))

    # Model, Loss, and Optimizer
    '''2-channel output: background / target area'''
    model = R2U_Net(img_ch=1, output_ch=2, t=1)
    # model = U_Net(img_ch=1, output_ch=2)
    # model = ResUnet(in_channel=1, out_channel=2)

    model = nn.DataParallel(model)
    model.to(device)

    replay_buffer = getbuffer(args, img_size, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[.9, .999], weight_decay=0.0)

    now = datetime.datetime.now()
    nowDate = now.strftime('%y%m%d_%H%M%S')
    ckptpath = os.path.join(args.ckptdir, nowDate)
    samplepath = os.path.join(args.sampledir, nowDate)
    start_epoch = 0

    if args.resumepath != '':
        file = args.resumepath + args.filename
        assert os.path.isfile(file)
        checkpoint = torch.load(file, map_location=device)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        replay_buffer = checkpoint["buffer"].cpu()  
        print("Loaded pretrained weights")
        # ckptpath = args.resumepath
        # samplepath = args.resume_gen

    # Main Iteration
    cur_iter = 0
    best_dice = 0.0
    save_epoch = 0

    new_lr = 0
    cur_lr = 0

    seg_weight = args.seg_weight
    energy_weight = args.energy_weight
    n_steps = args.n_steps

    diverge = False
    for epoch in range(start_epoch, args.epoch):
        if diverge: 
            break
        print("Epoch [%3d/%3d]  Energy weight %d  Seg Weight %d" % (epoch+1, args.epoch, energy_weight, seg_weight))

        if epoch == 30: 
            seg_weight = 100
            energy_weight = 1
            # n_steps = 60
        if epoch == 70: 
            seg_weight = 300
            energy_weight = 1
        # -------------------------------- Train Dataset --------------------------------
        model.train()
        for step, (data) in enumerate(train_loader):        
            cur_iter += 1
            image, label = data["image"], data["label"]
            
            pos_inputs = Variable(image).to(device).float()
            targets = Variable(label).to(device).float()

            total_loss = 0.
            
            pos_logits = model(pos_inputs) # 2*n*n logits
            # predict = output[:,1,:,:].unsqueeze(1)
            
            if seg_weight > 0:
                output = F.softmax(pos_logits, dim=1) # 2*n*n probabilities
                
                if args.loss == 'dice': 
                    seg_loss = dice_loss(predict, targets) # Dice loss
                else: 
                    seg_loss = F.cross_entropy(pos_logits, targets[:,0,:,:].long())

                _, ind = torch.max(output, dim=1)
                predict1 = (ind.unsqueeze(1) == 1) * 1.

                dice_score = dice_coef(predict1, (targets==1)*1.)
                
                total_loss += seg_loss * seg_weight
            
            energy_loss = 0.
            if energy_weight > 0:
                replay_buffer, neg_inputs = sample_q(args, model, img_size, replay_buffer, device, n_steps)
                
                neg_logits = model(neg_inputs)
                
                # Positive/Negative sample energy
                f_pos = pos_logits.logsumexp(1).sum((1,2)).mean()
                f_neg = neg_logits.logsumexp(1).sum((1,2)).mean()

                energy_loss = -(f_pos - f_neg)

                if abs(energy_loss) > 10000000000000000:
                    print("Model Diverged!")
                    diverge = True
                    break
                
                total_loss += energy_loss * energy_weight
                
                if (cur_iter % args.sample_freq) == 0:
                    os.makedirs(samplepath, exist_ok=True)
                    plot('{}/{:03d}_{:03d}.png'.format(samplepath, epoch+1, step), neg_inputs)
            
            if step % args.sample_freq ==0 :
                if (energy_weight > 0) and (seg_weight > 0):
                    print('     > Batch [%3d/%3d] Pos %.4f, Neg %.4f, Diff %.4f CE loss %.4f, Dice Coef %.4f' % (step+1, len(train_loader), f_pos, f_neg, f_pos-f_neg, seg_loss, dice_score))
                if (energy_weight > 0) and (seg_weight <= 0):
                    print('     > Batch [%3d/%3d] Pos %.4f, Neg %.4f, Diff %.4f' % (step+1, len(train_loader), f_pos, f_neg, f_pos-f_neg))
                if (energy_weight <= 0) and (seg_weight > 0):
                    print('     > Batch [%3d/%3d] CE loss %.4f, Dice Coef %.4f' % (step+1, len(train_loader), seg_loss, dice_score))
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # ----------------------------- Validation Dataset -----------------------------
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            val_dice = 0.
            n_samples = 0
            for step, (data) in enumerate(valid_loader):
                image, label = data["image"], data["label"]

                pos_inputs = Variable(image).to(device).float()
                targets = Variable(label).to(device).float()

                pos_logits = model(pos_inputs)
            
                output = F.softmax(pos_logits, dim=1)
                predict = output[:,1,:,:].unsqueeze(1)

                if args.loss == 'dice':
                    seg_loss = dice_loss(predict, targets)
                else: 
                    seg_loss = F.cross_entropy(pos_logits, targets[:,0,:,:].long())

                _, ind = torch.max(output, dim=1)
                predict1 = (ind.unsqueeze(1) == 1) * 1.
                dice_score = dice_coef(predict1, (targets==1)*1.)
                
                n_batch = int(image.size()[0])
                n_samples += n_batch
                val_loss += seg_loss.data * n_batch
                val_dice += dice_score.data * n_batch
                
            val_loss = val_loss / n_samples
            val_dice = val_dice / n_samples
            print('Validation: Dice loss %.4f Dice Coef %.4f\n' % (val_loss, val_dice))
        
        if val_dice >= best_dice :
            if epoch >20 :
                best_dice = val_dice
                state = {'model': model.state_dict(),
                            'epoch': epoch+1,
                            'buffer': replay_buffer,
                            'optimizer': optimizer.state_dict(),
                            'val_dice': val_dice}
                os.makedirs(ckptpath, exist_ok=True)
                torch.save(state, os.path.join(ckptpath, '{:03d}'.format(int(epoch+1))
                                                +'_{:05.4f}'.format(val_dice)+'.pt'))