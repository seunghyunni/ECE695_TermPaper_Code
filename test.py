import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.modified_unet import Modified2DUNet 
from utils.load_data_aug import HeartMRIDataset
import numpy as np
from sklearn.model_selection import train_test_split
from utils.metrics import dice_coef
from utils.losses import dice_loss, sensitivity, specificity, ppv, FN_dice, FP_dice
from utils.calibration import ece_function, sce_function
import os
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
import easydict
from model.network import R2U_Net
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from utils.ece import _ECELoss
from sklearn.calibration import calibration_curve
import math
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from tqdm import tqdm_notebook
import cv2
from sklearn.metrics import brier_score_loss


def get_pixel_num(arr):
    (unique, counts) = np.unique(arr, return_counts=True)
    if len(unique) == 1:
        return 0
    else:
        frequencies = np.asarray((unique, counts)).T
        return frequencies[1][1]


# Configuration
args = \
easydict.EasyDict({"crop_size": 128,
                   "batch_size": 1,
                   "lr": 1e-4,

                   "resumepath": './path_to_your_trained_model',
                   "filename": 'checkpoint name', 
                
                   "cal_bins": 10,
                   "energy_type":'logsumexp',
                   'save_dir':"./test_results/",
                   "box_width":10
                  })

torch.manual_seed(1)    
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
# Load Data
images = glob("./data_preprocessed/preprocessed_images/*.npy")
labels = glob("./data_preprocessed/preprocessed_labels/*.npy")

images.sort()
labels.sort()

test_images = images[1200:]
test_labels = labels[1200:]

test_images.sort()
test_labels.sort()

print(len(test_images), len(test_labels), test_images[0], test_labels[0]) 

# Dataloader
img_trans_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                # lambda x: x + args.sigma * torch.randn_like(x)
                                ])

# test img
img_trans_val = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

#mask
tar_trans = transforms.Compose([
                                transforms.ToTensor()
                                ])

test_data = HeartMRIDataset(test_images, test_labels, img_trans_train, img_trans_val, tar_trans, False)

test_loader = DataLoader(test_data, batch_size=args.batch_size,
                         shuffle=False, num_workers=2, drop_last=False)

img_size = test_data[0]['image'].size()
lab_size = test_data[0]['label'].size()
print("image size: {}".format(img_size))
print("label size: {}".format(lab_size))

# Load Model
'''2-channel output: background / target area'''
model = R2U_Net(img_ch=1, output_ch=2, t=1)

model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[.9, .999], weight_decay=0.0)

if args.resumepath != '':
    file = args.resumepath + args.filename
    print(file)
    # assert os.path.isfile(file)
    checkpoint = torch.load(file, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded pretrained weights")

def savefig(origin, target, confidence_map, pred, energy, ood_score, patient_id, file_id, dice_score):
    cm = plt.cm.get_cmap('jet')
    cm.set_under("1");cm.set_over("0")
    plt.rcParams['figure.figsize'] = [12, 3]

    fig, axs = plt.subplots(1,3)

    title = '(' + str(np.round_(ood_score.numpy()[0], 4)) + ")"
    ax1 = axs[0]
    ax1.set_title('GroundTruth ' + title)
    ax1.axis("off")
    ax1.imshow(origin, cmap = "gray")
    ax1.contour(target)

    ax2 = axs[1]
    title2 = '(' + str(np.round_(dice_score.data.cpu().detach().numpy(), 4)) + ")"
    ax2.set_title('Prediction ' + title2)
    ax2.axis("off")
    ax2.imshow(origin, cmap = "gray")
    ax2.contour(pred)

    ax3 = axs[2]
    title3 = '(' + str(np.int64(energy.cpu().detach().numpy())) + ")"
    ax3.set_title('Prob map ' + title3)
    ax3.axis("off")
    ax3.imshow(origin, cmap = "gray")
    c1 = ax3.imshow(confidence_map, cmap = cm, alpha=0.5)
    fig.colorbar(c1, ax=ax3)

    if not(os.path.exists(args.save_dir + "imgs/" + patient_id)):
        os.mkdir(args.save_dir + "imgs/" + patient_id)

    plt.savefig(args.save_dir +"imgs/"+ patient_id + "/{}_{}_{:.4f}.png".format(patient_id, file_id, dice_score.data), bbox_inches='tight')
    plt.cla()
    plt.close(fig)


def bounding_box(img):
    img = img.cpu().numpy()[0,0,:,:]
    cv_img = img.astype(np.uint8) 
    # threshold
    ret, thresh = cv2.threshold(cv_img, 0.5, 1, cv2.THRESH_BINARY)
    # find contour
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # draw surrounding rectangle
    try:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
    except:
        x = -1 
        y = -1 
        w = -1 
        h = -1
    return x,y,w,h


def condition(x,y,w,h,width, size = args.crop_size):
    if x-width>=0 and y-width >=0 and y+h+width <= size-1 and x+w+width <=size-1:
        return True
    else:
        return False


nll = nn.CrossEntropyLoss().cuda()


model.eval()
if True:
    dice_scores_list = []
    hd_list = []
    sens_list = []
    spec_list = []
    ppv_list = []

    test_loss = 0.
    test_dice = 0.
    test_hd = 0
    running_sens = 0
    running_spec = 0
    running_ppv = 0

    n_samples = 0

    df_energy = []
    df_slicenum = []
    df_patientid = []
    df_dice = []
    df_ece = []
    df_hd = []
    df_ood_score = []
    df_unc_score = []
    
    df_volume = []

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ece_criterion = _ECELoss(n_bins = args.cal_bins, device = device_)
    ece_fn = ece_function(n_bins = args.cal_bins, device = device)
    sce_fn = sce_function(n_bins = args.cal_bins, device = device)
    count = torch.zeros(args.cal_bins).to(device)
    conf = torch.zeros(args.cal_bins).to(device)
    acc = torch.zeros(args.cal_bins).to(device)
    nll = []
    brier = []
    ece = []
    
    for step, (data) in tqdm_notebook(enumerate(test_loader), total = len(test_loader)):
        image, label, patient_id, file_id = data["image"], data["label"], data["patient_id"][0], data["file_id"][0]
        patient_id = patient_id.split("/")[-1]
        #_, count = np.unique(label, return_counts=True)
        if True:
            pos_inputs = image.to(device).float()
            targets = label.to(device).float()
            
            pos_logits = model(pos_inputs)

            output = F.softmax(pos_logits, dim=1)
            #predict = output[:,1,:,:].unsqueeze(1)

            zeros = torch.zeros(output.size())
            ones = torch.ones(output.size())
            predict = torch.where(output.cpu() > 0.50, ones, zeros)  # threshold 0.50

            volume = get_pixel_num(predict[:,1,:,:])
            predict = Variable(predict).cuda()

            # for calibration
            # set foreground bounding box
            x, y, w, h = bounding_box(targets)
            if x == -1 and y == -1 and w == -1 and h == -1 : 
                continue
            
            width = args.box_width
            if condition(x,y,w,h,width, size=args.crop_size):
                # nll score
                nll_score = F.cross_entropy(pos_logits[:,:,x-width:x+w+width, y-width:y+h+width], 
                                            targets[:, 0, x-width:x+w+width, y-width:y+h+width].long(), reduction='mean')
                nll.append(nll_score.data)

                # brier score
                brier_score = brier_score_loss(targets[0,0,x-width:x+w+width, y-width:y+h+width].reshape(-1).cpu().detach().numpy(), 
                                               output[0,1,x-width:x+w+width, y-width:y+h+width].reshape(-1).cpu().detach().numpy())
                brier.append(brier_score)

                # ece score
                count_s, conf_s, acc_s = ece_fn(output[:,:,y-width:y+h+width,x-width:x+w+width],
                                            targets[:,:,y-width:y+h+width,x-width:x+w+width].long())
                count += count_s
                conf += conf_s
                acc += acc_s
                
                count_s = count_s.cpu().numpy()
                acc_s = acc_s.cpu().numpy()
                conf_s = conf_s.cpu().numpy()
                for i in range(args.cal_bins):
                    if count_s[i] == 0:
                        acc_s[i] == 0
                        conf_s[i] == 0
                    else:
                        acc_s[i] = acc_s[i]/count_s[i]
                        conf_s[i] = conf_s[i]/count_s[i]

                ece.append(np.sum((count_s/np.sum(count_s))*np.abs(acc_s-conf_s)))

            # Dice score
            dice_score = dice_coef(output[:,1,:,:].unsqueeze(1), targets)

            # Compute 95% tile Hausdorff Distance
            hd = max(np.percentile(directed_hausdorff(targets[0,0,:,:].cpu().detach().numpy(), predict[0,1,:,:].cpu().detach().numpy())[0], 95),
                    np.percentile(directed_hausdorff(predict[0,1,:,:].cpu().detach().numpy(), targets[0,0,:,:].cpu().detach().numpy())[0], 95))

            sens = sensitivity(predict[:, 1, :, :], targets[:, 0, :, :])
            spec = specificity(predict[:, 1, :, :], targets[:, 0, :, :])
            ppv_ = ppv(predict[:, 1, :, :], targets[:, 0, :, :])

            # Energy and OOD Score
            x_k = torch.autograd.Variable(pos_inputs, requires_grad=True).to(device)
            if args.energy_type == "l1":
                energy = -torch.abs(model(x_k)).sum(1).sum((1, 2))
            elif args.energy_type == "logsumexp":
                energy = -model(x_k).logsumexp(1).sum((1, 2))

            f_prime = torch.autograd.grad(-energy, [x_k], retain_graph=True)[0]
            grad = f_prime.view(1, -1)
            ood_score = -grad.norm(p=2, dim=1).detach().cpu()

            ######################################################################
            #                               Plot Figures                         #
            ######################################################################
            number = 0
            origin = pos_inputs.cpu().detach().numpy()[number, 0, :, :]
            target = targets.cpu().detach().numpy()[number, 0, :, :]
            pred = predict[:, 1, :, :].unsqueeze(1).cpu().detach().numpy()[number, 0, :, :]  # w, h
            confidence_map = output[:, 1, :, :].unsqueeze(1).cpu().detach().numpy()[number, 0, :, :]
            savefig(origin, target, confidence_map, pred, energy, ood_score, patient_id, file_id, dice_score)
            ######################################################################

            n_batch = int(image.size()[0])
            n_samples += n_batch
            test_dice += dice_score.data * n_batch
            test_hd += hd * n_batch

            # AUC
            if sens != 0:
                running_sens += sens.data * n_batch
                sens_list += [sens.data] * n_batch

            running_spec += spec.data * n_batch
            spec_list += [spec.data] * n_batch

            running_ppv += ppv_.data * n_batch
            ppv_list += [ppv_.data] * n_batch

            dice_scores_list += [dice_score.data] * n_batch
            hd_list += [hd] * n_batch
            
            # Uncertainty Estimation
            index = (predict[0,1,:,:].cpu().numpy()) == 1
            prob = output.cpu().detach().numpy()
            prob1 = prob[0,0,:,:][index]
            prob2 = prob[0,1,:,:][index]
            uncertainty_score = -np.mean(prob1 * np.log(prob1) + prob2 * np.log(prob2))
            
            # Record results            
            df_patientid.append(patient_id)
            df_slicenum.append(file_id)
            df_dice.append(dice_score.data.cpu().detach().numpy())
            df_energy.append(energy.cpu().detach().numpy()[0])
            df_hd.append(hd)
            df_ood_score.append(ood_score.numpy()[0])
            df_unc_score.append(uncertainty_score)
            
            df_volume.append(volume * 0.001)

            if (step == 0) or (step + 1) % 1 == 0:
                print('     > Step [%3d/%3d] Dice Coef %.4f, 95 Hausdorff Distance %.4f' % (step + 1, len(test_loader), dice_score.data, hd))
                if sens != 0:
                    print('                      Sensitivity %.4f, Specificity %.4f, ppv %.4f' % (
                    sens.data, spec.data, ppv_.data))

    test_loss = test_loss / n_samples
    test_dice = test_dice / n_samples
    test_hd = test_hd / n_samples

    test_spec = running_spec / n_samples
    test_ppv = running_ppv / n_samples

    mean_hd = np.mean(np.array(df_hd))
    std_hd = np.std(np.array(df_hd))
    print("Mean, std of Hausdorff Distance ", mean_hd, std_hd)
    
    mean_dice = torch.mean(torch.stack(dice_scores_list))
    std_dice = torch.std(torch.stack(dice_scores_list))
    print("Mean, std of Dice ", mean_dice, std_dice)


    mean_sens = torch.mean(torch.stack(sens_list))
    std_sens = torch.std(torch.stack(sens_list))
    std_spec = torch.std(torch.stack(spec_list))
    std_ppv = torch.std(torch.stack(ppv_list))

    print("Total Sensitivity %.4f , std %.4f" % (mean_sens, std_sens))
    print("Total Specificity %.4f , std %.4f" % (test_spec, std_spec))
    print("Total PPV %.4f , std %.4f" % (test_ppv, std_ppv))
    
    # OOD Scores
    mean_s1 = np.mean(np.array(df_energy))
    std_s1 = np.std(np.array(df_energy))
    
    mean_s2 = np.mean(np.array(df_ood_score))
    std_s2 = np.std(np.array(df_ood_score))
    
    mean_s3 = np.mean(np.array(df_unc_score))
    std_s3 = np.std(np.array(df_unc_score))
    
    print("S1 Score %.4f, std %.4f" %(mean_s1, std_s1))
    print("S2 Score %.4f, std %.4f" %(mean_s2, std_s2))
    print("S3 Score %.4f, std %.4f" %(mean_s3, std_s3))
    
    # Calibration Scores
    # ECE
    count = count.cpu().numpy()
    acc = acc.cpu().numpy()
    conf = conf.cpu().numpy()
    for i in range(args.cal_bins):
        if count[i] == 0:
            acc[i] == 0
            conf[i] == 0
        else:
            acc[i] = acc[i]/count[i]
            conf[i] = conf[i]/count[i]
    
    ece_ = np.sum((count/np.sum(count))*np.abs(acc-conf))
    
    mean_ece = np.mean(np.array(ece))
    std_ece = np.std(np.array(ece))
    
    # Brier
    mean_brier = np.mean(np.array(brier))
    std_brier = np.std(np.array(brier))
    
    # NLL
    mean_nll = torch.mean(torch.stack(nll))
    std_nll = torch.std(torch.stack(nll))
    
    print("")
    print("ece Score: %.4f" %(ece_))
    print("ece2 Score: %.4f, std %.4f " %(mean_ece, std_ece))
    print("Brier Score: %.4f, std %.4f " %(mean_brier, std_brier))
    print("NLL Score: %.4f, std %.4f " %(mean_nll.data, std_nll.data))



with torch.no_grad():
    ################## Reliability Diagram ###################
    
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = [3, 3]

    # only these two lines are calibration curves
    plt.bar(conf, acc, width=0.05)
    plt.plot(conf[5:], acc[5:], marker='o', linewidth=1, label='JEM', color = 'red')

    # reference line, legends, and axis labels
    line = mlines.Line2D([0.0, 1.0], [0.0, 1.0], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    plt.title('Reliability Diagram')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')

    plt.tight_layout()
    plt.legend()
        
    plt.savefig("{}/reliability_diagram.png".format(args.save_dir))
    
    ############################################################

    # Change to Dataframe and save as csv file
    output_df = df(np.array([df_patientid, df_slicenum, df_dice, df_hd, df_energy, df_ood_score, df_unc_score, df_volume]).T,
                    columns=['patient_id', 'slice_num', 'dice', 'Hausdorff_Distance', 'energy', 'ood_score', 'uncertainty_score', 'volume'])
    output_df.to_csv("{}result.csv".format(args.save_dir))

    print("Completed saving csv file")

    print("\n============== Points for each Bin =================")
    for i, cnt in enumerate(count):
        print("%d: %d" % (i, cnt))
    
    ############################################################
    
#     f = open("{}/result.txt".format(args.save_dir), 'w')
#     f.write("Performance\n")
#     f.write(" >> Dice Coefficient: %.4f, std %.4f\n" %(mean_dice, std_dice))
#     f.write(" >> 95%% Hausdorff Distance: %.4f, std %.4f\n" %(mean_hd, std_hd))
    
#     f.write("\nCalibration Result\n")
#     f.write(" >> ECE Score: %.4f, std %.4f\n" %(mean_ece, std_ece))
#     f.write(" >> Brier Score: %.4f, std %.4f\n" %(mean_brier, std_brier))
#     f.write(" >> NLL Score: %.4f, std %.4f\n" %(mean_nll.data, std_nll.data))
    
#     f.write("\nOOD Score Result\n")
#     f.write(" >> S1 Score %.4f, std %.4f\n" %(mean_s1, std_s1))
#     f.write(" >> S2 Score %.4f, std %.4f\n" %(mean_s2, std_s2))
#     f.write(" >> S3 Score %.4f, std %.4f" %(mean_s3, std_s3))
    
#     f.close()
