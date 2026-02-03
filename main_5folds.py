import os
import csv
import random
import shutil
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as mds
from data.dataloaders_new import CustomDataloader_patient_sample as Dataloader
import yaml
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report, cohen_kappa_score, precision_score,roc_curve
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Load config
# ==============================
with open('config_loss.yaml', 'rb') as f:
    config = yaml.safe_load(f.read())

image_size = config["train"]["size"]
resume = config["train"]["resume"]
random_seed = config["train"]["seed"]
basic_learning_rate = config["train"]["lr"]
batch_size = config["train"]["batch-size"]
epochs = config["train"]["epochs"]
decay = config["train"]["decay"]
data_path = config["train"]["data_path"]
num_worker = config["train"]["num_worker"]
num_class = config["train"]["num_classes"]
bool_pretrain = config["train"]["pretrain"]
mixed_loss_bool = config["train"]["mixed_loss"]

loss_file = config["model"]["loss_file"]
loss_file_mix = config["model"]["loss_file_mix"]
loss_caller = config["model"]["loss_caller"]
loss_caller_mix = config["model"]["loss_caller_mix"]
loss_alpha = config["model"]["alpha"]
loss_beta = config["model"]["beta"]
caller = config["model"]["caller"]
model_name = config["model"]["name"]
file_name = config["model"]["file"]
checkpoint_folder = config["model"]["folder"]
model_pretrain = config["model"]["pretrain"]

log_folder = config["log"]["folder"]

# ==============================
# Seed
# ==============================
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
cudnn.deterministic = True

# ==============================
# Create folders
# ==============================
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)
shutil.copyfile('config_loss.yaml', os.path.join(log_folder, 'config.log'))
log_file = os.path.join(log_folder, model_name + '_' + str(random_seed) + '_freeze0.csv')
# ==============================
# Loss and Model Import
# ==============================
exec('from losses.{} import *'.format(loss_file))
exec('from losses.{} import *'.format(loss_file_mix))
exec('from models.{} import *'.format(file_name))

use_cuda = torch.cuda.is_available()

# ==============================
# Helper Functions (train/valid/test)
# ==============================
def save_checkpoint(model, acc, epoch, fold):
    state = {
        'net': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(checkpoint_folder, f"{model_name}_fold{fold}_best_new.pth"))

def adjust_learning_rate(optimizer, epoch):
    lr = basic_learning_rate
    if epoch % 20 == 0:
        lr /= 5
    if epoch > 100:
        lr = 0.00035
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)


        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(loader), f"Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})")
    return train_loss / len(loader), 100.*correct/total

def validate(model, loader, criterion, epoch, fold):
    model.eval()
    valid_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if mixed_loss_bool:
                loss1 = criterion[0](outputs, targets)
                loss2 = criterion[1](outputs, targets)
                loss = loss_alpha * loss1 + loss_beta * loss2
            else:
                loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(loader), f"Val Loss: {valid_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})")
        # Save checkpoint.
    acc = 100. * correct / total
    print('Epoch: %d' % epoch, 'acc: %.3f%%' % acc)

    return valid_loss / batch_idx, 100. * correct / total
import time  # 新增
def test_model(fold_idx, criterion,  roc_dir):

    model = eval(f'{caller}(num_classes=num_class)')
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model_path = os.path.join(checkpoint_folder,
                              f'{model_name}_fold{fold_idx}_best_new.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])

    model.eval()
    y_true_list = []
    y_pred_list = []
    y_score_list = []

    total_loss = 0.0

    # dataloader: 用 val_loader 或 test_loader
    dataloader = Dataloader(batch_size=batch_size, num_workers=num_worker,
                            img_resize=image_size,
                            root_dir=os.path.join(data_path, str(fold_idx)))
    test_loader, _ = dataloader.run("val")  # 或 "test" 根据数据集实际情况

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # loss
            if mixed_loss_bool:
                loss1 = criterion[0](outputs, targets)
                loss2 = criterion[1](outputs, targets)
                loss = loss_alpha * loss1 + loss_beta * loss2
            else:
                loss = criterion(outputs, targets)
            total_loss += loss.item()

            #
            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            if num_class == 2:
                y_score_list.extend(probs[:, 1].cpu().numpy())  #

    #  numpy
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_score = np.array(y_score_list) if num_class == 2 else None

    #
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='binary' if num_class==2 else 'macro')
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    #
    test_pre = precision_score(
        y_true, y_pred,
        average='binary' if num_class == 2 else 'macro',
        zero_division=0
    )
    if num_class == 2:
        test_auc = roc_auc_score(y_true, y_score)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        test_sen = tp / (tp + fn + 1e-8)
        test_spe = tn / (tn + fp + 1e-8)

        # ROC CSV
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        roc_csv_path = os.path.join(roc_dir, f"{model_name}_fold{fold_idx}_roc_new.csv")
        roc_df.to_csv(roc_csv_path, index=False)
        print(f"Fold {fold_idx} ROC CSV saved to: {roc_csv_path}")
    else:
        test_auc = None
        test_sen = None
        test_spe = None
        test_pre = None


    print(f"Fold {fold_idx} test run time: {run_time:.2f} seconds")
    return test_acc, test_bal_acc, test_f1, test_auc, test_pre,  test_sen, test_spe, y_true, y_score

# ==============================
# Main Training Loop (Five-Fold)
# ==============================
# ==============================
# Main Training Loop with Unified Logging (Five-Fold)
# ==============================
def main():
    global lws
    # CSV
    log_file = os.path.join(log_folder, f"{model_name}_five_fold_log_new.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold", "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "test_acc", "test_bal_acc", "test_f1", "test_auc","test_pre", "test_sen", "test_spe"
            ])

    fold_metrics = []
    roc_dir = os.path.join(log_folder, "roc")
    os.makedirs(roc_dir, exist_ok=True)

    for fold_idx in range(5):
        print(f"\n=== Fold {fold_idx} ===")
        dataloader = Dataloader(
            batch_size=batch_size,
            num_workers=num_worker,
            img_resize=image_size,
            root_dir=os.path.join(data_path, str(fold_idx))
        )

        train_loader, _ = dataloader.run("train")
        val_loader, _ = dataloader.run("val")
        #
        print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
        # Model
        model = eval(f'{caller}(num_classes=num_class)')
        if bool_pretrain:
            premodel_dict = getattr(mds, model_pretrain)(weights='IMAGENET1K_V1')
            pretrain_dict = premodel_dict.state_dict()
            model_dict = model.state_dict()
            select_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict and not k.startswith('fc')}
            model_dict.update(select_dict)
            model.load_state_dict(model_dict)

        model = nn.DataParallel(model).to(device)

        # Loss
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=basic_learning_rate, momentum=0.9, weight_decay=decay)

        # Training
        best_acc = 0.0
        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch)
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, epoch, fold_idx)

            #
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([fold_idx, epoch, train_loss, train_acc, val_loss, val_acc,
                                 "", "", "", "", "", ""])
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, best_acc, epoch, fold_idx)
        #
        test_acc, test_bal_acc, test_f1, test_auc,test_pre, test_sen, test_spe , y_true, y_score= test_model(fold_idx, criterion,roc_dir)
        fold_metrics.append([test_acc, test_bal_acc, test_f1, test_auc, test_pre,test_sen, test_spe])

            #
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([fold_idx, "best_model", "", "", "", "",
                             test_acc, test_bal_acc, test_f1, test_auc,test_pre, test_sen, test_spe])
            #
        if num_class == 2:
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
                roc_csv_path = os.path.join(roc_dir, f"{model_name}_fold{fold_idx}_roc_new.csv")
                roc_df.to_csv(roc_csv_path, index=False)
                print(f"Fold {fold_idx} ROC CSV saved to: {roc_csv_path}")

    fold_metrics = np.array(fold_metrics)
    mean_metrics = np.nanmean(fold_metrics, axis=0)
    std_metrics = np.nanstd(fold_metrics, axis=0)
    metric_names = ["ACC", "Balanced_ACC", "F1", "AUC", "PRE","SEN", "SPE"]

    print("\n=== Five-Fold Summary ===")
    for i, name in enumerate(metric_names):
        print(f"{name}: {mean_metrics[i]:.5f} ± {std_metrics[i]:.5f}")

    # 保存五折统计
    summary_file = os.path.join(log_folder, f"{model_name}_five_fold_summary_new.csv")
    pd.DataFrame({
        "metric": metric_names,
        "mean": mean_metrics,
        "std": std_metrics
    }).to_csv(summary_file, index=False)
    print("Five-fold summary saved to:", summary_file)


if __name__ == "__main__":
    main()
