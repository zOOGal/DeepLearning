import argparse
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import glob
import torchvision.transforms as tvt
import torch
import os


class CatDogDataset(Dataset):
    def __init__(self, root, transform, train_or_test, class_list):
        """
        label_array
        0 -- cat
        1 -- dog
        """
        self.root = root
        self.labels = list()
        self.train_or_test = train_or_test
        self.data_list = list()
        self.transform = transform
        self.class_list = class_list

        if train_or_test == 'train':
            self.path = os.path.join(self.root, 'Train')
        elif train_or_test == 'test':
            self.path = os.path.join(self.root, 'Val')

        for img_name in glob.glob(self.path + '/*/*/*', recursive=True):
            img_data = Image.open(os.path.join(self.path, img_name))
            # print(img_data)
            base_folder = os.path.basename(os.path.dirname(os.path.join(self.path, img_name)))
            # print()
            if self.class_list[0][1:-1] in str(base_folder):
                self.labels.append([1, 0])
            elif self.class_list[1][1:-1] in str(base_folder):
                self.labels.append([0, 1])

            self.data_list.append(img_data)

        self.labels = np.asarray(self.labels)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img = self.data_list[idx]
        corr_label = self.labels[idx]
        # transform image if transform exists
        if self.transform:
            img = self.transform(img)
        return img, corr_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HW02 Task2')
    parser.add_argument('--imagenet_root', type=str, required=True)
    parser.add_argument('--class_list', nargs='*', type=str, required=True)
    args, args_other = parser.parse_known_args()

    root = args.imagenet_root
    class_list = args.class_list

    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CatDogDataset(root, transform=transform, train_or_test='train', class_list=class_list)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=4)

    val_dataset = CatDogDataset(root, transform=transform, train_or_test='test', class_list=class_list)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True, num_workers=4)

    dtype = torch.float64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 40
    D_in, H1, H2, D_out = 3 * 64 * 64, 1000, 256, 2
    w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
    w2 = torch.randn(H1, H2, device=device, dtype=dtype)
    w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
    learning_rate = 1e-9

    for t in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            x = inputs.view(inputs.size(0), -1)
            h1 = x.mm(w1.float())
            h1_relu = h1.clamp(min=0)
            h2 = h1_relu.mm(w2.float())
            h2_relu = h2.clamp(min=0)
            y_pred = h2_relu.mm(w3.float())
            y = labels.view(labels.size(0), -1)

            loss = (y_pred - y).pow(2).sum().item()
            y_error = y_pred - y
            epoch_loss += loss

            grad_w3 = h2_relu.t().mm(2 * y_error)
            h2_error = 2.0 * y_error.mm(w3.t().float())
            h2_error[h2 < 0] = 0
            grad_w2 = h1_relu.t().mm(2 * h2_error)
            h1_error = 2.0 * h2_error.mm(w2.t().float())
            h1_error[h1 < 0] = 0
            grad_w1 = x.t().mm(2 * h1_error)

            # Update weights using gradient descent
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
            w3 -= learning_rate * grad_w3

        # print loss per epoch
        print('Epoch %d:\t %0.4f' % (t, epoch_loss))
        # save output to txt file
        f = open("output.txt", "a")
        print('Epoch %d:\t %0.4f' % (t, epoch_loss), file=f)
        # f.write('Epoch%d:\t%0.4f' % (t, epoch_loss))
        f.close()

    # Store layer weights in pickle file format
    torch.save({'w1': w1, 'w2': w2, 'w3': w3}, './wts.pkl')

    # load weights from training result
    weights = torch.load('./wts.pkl')
    w1_val = weights['w1']
    w2_val = weights['w2']
    w3_val = weights['w3']

    correct = 0  # for val accuracy
    total = 0
    loss = 0
    for i, data in enumerate(val_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        x = inputs.view(inputs.size(0), -1)
        h1 = x.mm(w1_val.float())
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2_val.float())
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3_val.float())

        # Compute loss
        y = labels.view(labels.size(0), -1)
        # loss for validation set
        temp_loss = (y_pred - y).pow(2).sum().item()
        loss += temp_loss
        # accuracy for validation set
        # ref:https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch
        _, predicted = torch.max(y_pred, 1)
        # print(predicted)
        for i in range(10):  # batchsize
            total += 1
            if predicted[i] == y[i][1]:  # 0: [1,0] 1:[0,1]
                correct += 1
            else:
                pass
    acc = correct / total * 100
    # print
    print('Val Loss: %0.4f' % loss)
    print('Val Accuracy:%0.4f%%' % acc)

    # save to .txt in append mode
    f = open("output.txt", "a")
    # f.write('Val Loss: %0.4f' % loss)
    print('', file=f)
    print('Val Loss: %0.4f' % loss, file=f)
    print('Val Accuracy:%0.4f%%' % acc, file=f)
    f.close()
