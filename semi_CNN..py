# [Note] This file is copied from https://github.com/1am9trash/HUNG_YI_LEE_ML_2021/blob/main/hw/hw3/hw3_code.ipynb

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder

# * what is this 
# ! danger
# ? is it correct
# TODO: hahaha, great work!
# FIXME: 
# @param 

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        
        return x


class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id]

def get_pseudo_labels(dataset, model, threshold=0.9):
    '''
    If an image in unlabelled set has high probability (>threshold) to possess one label, then treat this image as if it is actually labelled that way
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    idx = []
    labels = []

    for i, batch in enumerate(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)

        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * BATCH_SIZE + j)
                labels.append(int(torch.argmax(x)))

    model.train()
    print ("\nNew data: {:5d}\n".format(len(idx)))
    dataset = PseudoDataset(Subset(dataset, idx), labels)
    return dataset


if __name__=="__main__":
    # data augmentation
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop((128, 128)),
        transforms.RandomChoice(
            [transforms.AutoAugment(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
        transforms.ToTensor(),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])




    BATCH_SIZE = 15  # fixme: original 128

    # datasets.
    train_set = DatasetFolder("/home/prp/fra/cnn_hwk/food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder("/home/prp/fra/cnn_hwk/food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder("/home/prp/fra/cnn_hwk/food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder("/home/prp/fra/cnn_hwk/food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    # data loaders.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model
    model = Classifier().to(device)
    model.device = device

    # Call resnet-18
    # model = torchvision.models.resnet18(pretrained=False).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 11).to(device)
    # model.device = device

    # Read pre-trained model
    # model = Classifier().to(device)
    # model.load_state_dict(torch.load("model.ckpt"))
    # model.device = device


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    N_EPOCHS = 5
    DO_SEMI = True
    model_path = "model.ckpt"

    best_acc = 0.0 #acc=accuracy
    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []

    for epoch in range(N_EPOCHS):
        if DO_SEMI and best_acc > 0.7 and epoch % 5 == 0:
            pseudo_set = get_pseudo_labels(unlabeled_set, model)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        # ---------- Train ----------
        model.train()

        train_loss = []
        train_accs = []

        for batch in train_loader:
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d} / {N_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in valid_loader:
            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d} / {N_EPOCHS:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # ---------- Record ----------
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
        train_loss_record.append(train_loss)
        valid_loss_record.append(valid_loss)
        train_acc_record.append(train_acc)
        valid_acc_record.append(valid_acc)

        # ---------- Test ----------
        # model.eval()

        # predictions = []

        # for batch in test_loader:
        #     imgs, labels = batch

        #     with torch.no_grad():
        #         logits = model(imgs.to(device))

        #     predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

        # with open("predict.csv", "w") as f:
        #     f.write("Id,Category\n")

        #     for i, pred in  enumerate(predictions):
        #         f.write(f"{i},{pred}\n")

