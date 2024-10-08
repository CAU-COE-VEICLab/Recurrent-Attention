import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import imageio
import os
from sklearn.metrics import f1_score


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        if self.n_classes == 1:
            temp_prob = input_tensor == 1  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        else:
            for i in range(self.n_classes):
                temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
                tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        N = target.size()[0]
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        if self.n_classes == 1:
            loss = 1 - loss.sum()/N
        else:
            loss = 1 - loss
        #print('a')
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            if self.n_classes == 1:
                inputs = torch.sigmoid(inputs)
            else:
                inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        if self.n_classes == 1:
            dice = self._dice_loss(inputs.squeeze(1), target.squeeze(1))
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[0]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss_guorun(nn.Module):
    def __init__(self):
        super(DiceLoss_guorun, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss_guorun()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


def test_single_2Dmedimage(image, label, net, classes, save_image_name, test_save_path=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # print(image.shape)

    input = torch.from_numpy(image).unsqueeze(
        0).float().cuda()
    B, C, H, W = input.shape
    net.eval()
    with torch.no_grad():
        if classes == 1:
            out = torch.sigmoid(net(input)).squeeze(0).cpu()  # Sigmod回归后去掉批次维N
            prediction = np.where(np.array(out) > 0.5, 1, 0)  # 阈值默认为0.5
        else:
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    if classes == 1:
        classes += 1
    for i in range(1, classes):
        metric_list.append(calculate_metric_2Dimage_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_2D = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        img_2D[:, :, 0][np.where(prediction == 1)] = 255
        img_2D[:, :, 1][np.where(prediction == 2)] = 255
        img_2D[:, :, 2][np.where(prediction == 3)] = 255
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
        imageio.imwrite(os.path.join(test_save_path, save_image_name + '.png'), img_2D)

    return metric_list


def calculate_metric_2Dimage_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    iou = metric.binary.jc(pred, gt)
    precision_score = metric.binary.precision(pred, gt)
    recall_score = metric.binary.recall(pred, gt)
    # f1 = f1_score(gt, pred, average='binary')
    return dice, hd95, iou, precision_score, recall_score, 1  # f1
