import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Define the Args class to store training parameters
class Args:
    def __init__(self, batch_size=4, lr=0.001, epochs=20, device=None, save_path='save.pth',
                 beat_path='unet_best.pth', resume=False, resume_path='read.pth'):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.beat_path = beat_path
        self.resume = resume
        self.resume_path = resume_path

#data
def data_devide(images_path, labels_path, train_ratio=0.7, select_k = [1,2], select_block = [9,9], select = False):
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    dirpaths = ['dataset/train', 'dataset/train_label', 'dataset/test', 'dataset/test_label']
    for dirpath in dirpaths:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)

    # Get all image and label file names
    total_images = [f for f in os.listdir(images_path) if f.lower().endswith(IMAGE_EXTS)]
    total_names = [os.path.splitext(f)[0] for f in total_images]
    total_exts  = [os.path.splitext(f)[1] for f in total_images]
    num = len(total_names)
    indices = list(range(num))

    train_indices = random.sample(indices, int(num * train_ratio))
    test_indices = list(set(indices) - set(train_indices))

    def is_valiad(number, select_k, select_block):
        for k,block in zip(select_k,select_block):
            if (int(number) - k) % block == 0:
                return False
        return True

    if select:
        for i in train_indices:
            name = total_names[i]
            ext  = total_exts[i]
            if is_valiad(name,select_k,select_block):
                shutil.copy(os.path.join(images_path, name + ext), os.path.join('dataset/train', name + ext))
                shutil.copy(os.path.join(labels_path, name + ext), os.path.join('dataset/train_label', name + ext))
        for i in test_indices:
            name = total_names[i]
            ext  = total_exts[i]
            if is_valiad(name,select_k,select_block):
                shutil.copy(os.path.join(images_path, name + ext), os.path.join('dataset/test', name + ext))
                shutil.copy(os.path.join(labels_path, name + ext), os.path.join('dataset/test_label', name + ext))

    else:
        for i in train_indices:
            name = total_names[i]
            ext  = total_exts[i]
            shutil.copy(os.path.join(images_path, name + ext), os.path.join('dataset/train', name + ext))
            shutil.copy(os.path.join(labels_path, name + ext), os.path.join('dataset/train_label', name + ext))
        for i in test_indices:
            name = total_names[i]
            ext  = total_exts[i]
            shutil.copy(os.path.join(images_path, name + ext), os.path.join('dataset/test', name + ext))
            shutil.copy(os.path.join(labels_path, name + ext), os.path.join('dataset/test_label', name + ext))

def data_read(visual = True, vis_num = 5):
    train_image = os.listdir('dataset/train/')
    trans = torchvision.transforms.ToTensor()
    images = []
    labels = []
    for i in train_image:
        image_path = 'dataset/train/' + i
        label_path = 'dataset/train_label/' + i
        # print(image_path)
        image_i = Image.open(image_path)
        image_i = trans(image_i)
        label_i = Image.open(label_path)
        label_i = trans(label_i)
        images.append(image_i)
        labels.append(label_i)

    test_image = os.listdir('dataset/test/')
    images_test = []
    labels_test = []
    for i in test_image:
        image_path = 'dataset/test/' + i
        label_path = 'dataset/test_label/' + i
        image_i = Image.open(image_path)
        image_i = trans(image_i)
        label_i = Image.open(label_path)
        label_i = trans(label_i)
        images_test.append(image_i)
        labels_test.append(label_i)

    print('Train data: {} samples, shape: {}'.format(len(images), images[0].shape))
    print('Test data: {} samples, shape: {}'.format(len(images_test), images_test[0].shape))

    if visual:
        fig = plt.figure(figsize=(25, 8))
        for i in range(1, vis_num):
            ax = fig.add_subplot(2, vis_num, i)
            ax.imshow(images[i - 1].reshape(256, 256), cmap='gray')
            ax.set_title('Augmented image' + str(i))
            ax.grid(alpha=0.5)
            ax = fig.add_subplot(2, vis_num, i + vis_num)
            # ax.imshow(labels[i-1])
            ax.imshow(labels[i - 1].reshape(256, 256), interpolation='Gaussian', cmap='gray')
            ax.set_title('Ground truth' + str(i))
            ax.grid(alpha=0.75)
        plt.show()

    return images,labels,images_test,labels_test

# Define the GetLoader class, inheriting from Dataset,
# and override __getitem__() and __len__() methods
class GetLoader(torch.utils.data.Dataset):
    # __init__() method initializes the dataset
    def __init__(self, image_path, label_path, if_train=True):
        image_dir = os.listdir(image_path)
        image_name_list = [row.split('.')[0] for row in image_dir]
        image_path_list = []
        num = len(image_name_list)
        for i in range(num):
            image_i_path = image_path + image_name_list[i] + '.png'
            image_path_list.append(image_i_path)
        self.image_path_list = image_path_list

        label_dir = os.listdir(label_path)
        label_name_list = [row.split('.')[0] for row in label_dir]
        label_path_list = []
        num = len(label_name_list)
        for i in range(num):
            label_i_path = label_path + label_name_list[i] + '.png'
            label_path_list.append(label_i_path)
        self.label_path_list = label_path_list

        self.transform_image = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize(0.5, 0.5)])
        self.transform_label = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                               torchvision.transforms.ToTensor()])

        self.train_transform = [torchvision.transforms.RandomRotation(90),  # random rotation
                                torchvision.transforms.RandomVerticalFlip(p=0.5),  # random vertical flip
                                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                torchvision.transforms.RandomResizedCrop(256, (0.5, 1))  # random crop
                                ]
        self.add_noise = ["gaussian", "poisson", "salt", "speckle"]

        self.if_train = if_train

    # index is the index obtained after dividing data by batch size;
    # finally, return data and corresponding labels together
    def __getitem__(self, index):
        image_index_path = self.image_path_list[index]
        label_index_path = self.label_path_list[index]

        image = Image.open(image_index_path).convert("L")
        label = Image.open(label_index_path).convert("L")

        image = self.transform_label(image)
        label = self.transform_label(label)

        return image, label

    # This function returns the size/length of the data to facilitate DataLoader batching
    def __len__(self):
        return len(self.image_path_list)

def data_load(args):
    # Load data using GetLoader, returning a Dataset object containing data and labels
    image_train_path = 'dataset/train/'
    label_train_path = 'dataset/train_label/'
    image_test_path = 'dataset/test/'
    label_test_path = 'dataset/test_label/'
    train_loader = GetLoader(image_train_path, label_train_path, if_train=True)
    test_loader = GetLoader(image_test_path, label_test_path, if_train=False)

    train_Loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_Loader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=True)

    return train_Loader,test_Loader

class SimpleSegmentationMetric:
    def __init__(self):
        self.confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    def _gen_confusion_matrix(self, pred, label):
        mask = (label >= 0) & (label < 2)
        label = 2 * label[mask] + pred[mask]
        count = np.bincount(label, minlength=4)
        return count.reshape(2, 2)

    def add(self, preds, labels):
        # expects (B, 1, H, W) or (B, H, W)
        if preds.ndim == 4:
            preds = preds[:, 0, :, :]
            labels = labels[:, 0, :, :]
        for pred, label in zip(preds, labels):
            assert pred.shape == label.shape
            self.confusion_matrix += self._gen_confusion_matrix(pred, label)

    def pixel_accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def mean_iou(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        union = TP + FP + FN
        IoU = TP / np.maximum(union, 1e-10)  # 防止除以 0
        return np.mean(IoU)

    def reset(self):
        self.confusion_matrix = np.zeros((2, 2), dtype=np.int64)

def trans(data):
    data = np.array(data)
    data = data.astype('float32')
    data = torch.tensor(data)
    return data

def train(net, train_Loader, test_Loader, args):
    model = net.cuda()
    print("Model device:", next(model.parameters()).device)

    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,150], gamma=0.1)

    train_epoch_loss = []
    val_epoch_loss = []
    min_accuracy = 0
    train_accuracy = []
    test_accuracy = []
    train_miou = []
    test_miou = []

    # Restore training state
    if args.resume:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Start epoch:", epoch + 1)

    # start train
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_accuracy_i = 0
        train_miou_i = 0
        since_time = time.time()
        metric = SimpleSegmentationMetric()   # The parentheses indicate the number of classes
        for image, label in train_Loader:
            # for batch_idx,() in enumerate(train_Loader):
            image, label = trans(image), trans(label)
            image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()   # Clear gradients before using
            output = model(image)
            #output = torch.softmax(output, dim = 1)
            # print(output.max(),output.min())  # shape: (B, C, H, W)
            loss = Loss(output, torch.squeeze(label).long())
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # Get predicted class: select the class with the highest probability
            pred = torch.argmax(output, dim=1)  # shape: (B, H, W)
            # print(pred.max(), pred.min())
            label_np = label.squeeze(1).cpu().numpy().astype(np.int64)   # shape: (B, H, W)
            pred_np = pred.cpu().numpy().astype(np.int64)  # shape: (B, H, W)
            # Add to metric
            metric.add(pred_np, label_np)
            train_accuracy_i += metric.pixel_accuracy()
            train_miou_i += metric.mean_iou()

        # Evaluation mode: no gradient backpropagation needed
        # Test accuracy on the test dataset
        model.eval()
        val_loss = []
        test_accuracy_i = 0
        test_miou_i = 0
        with torch.no_grad():
            for image, label in test_Loader:
                image, label = trans(image), trans(label)
                image, label = image.cuda(), label.cuda()
                output = model(image)
                #output = torch.softmax(output, dim = 1)
                loss = Loss(output, torch.squeeze(label).long())
                val_loss.append(loss.item())

                # Save evaluation parameters
                pred = torch.argmax(output, dim=1)  # shape: (B, H, W)
                label_np = label.squeeze(1).cpu().numpy().astype(np.int64)  # shape: (B, H, W)
                pred_np = pred.cpu().numpy().astype(np.int64)  # shape: (B, H, W)
                # Add to metric
                metric.add(pred_np, label_np)
                test_accuracy_i += metric.pixel_accuracy()
                test_miou_i += metric.mean_iou()

        # Save model parameters
        if test_accuracy_i / len(test_Loader) > min_accuracy:
            # Save the best-performing model
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                args.beat_path
            )
            print("best accuracy has saved:{:.4f} --- > {:.4f}".format(min_accuracy,
                                                                       (test_accuracy_i / len(test_Loader))))
            min_accuracy = test_accuracy_i / len(test_Loader)

        #The last epoch save
        if epoch + 1 == args.epochs:
            # save model
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                args.save_path
            )
            print("train over")


        train_accuracy.append(train_accuracy_i / len(train_Loader))
        test_accuracy.append(test_accuracy_i / len(test_Loader))
        train_miou.append(train_miou_i / len(train_Loader))
        test_miou.append(test_miou_i / len(test_Loader))

        train_loss = torch.tensor(train_loss, device='cpu')
        train_loss_ave = np.average(train_loss)
        train_epoch_loss.append(train_loss_ave)

        val_loss = torch.tensor(val_loss, device='cpu')
        val_loss_ave = np.average(val_loss)
        val_epoch_loss.append(val_loss_ave)

        # Learning rate optimization
        scheduler.step()
        print('Learning rate: {:.6f}'.format(scheduler.get_last_lr()[0]))

        if (epoch) % 1 == 0:
            print('-------------------Epoch:{}/{}\t ------------------\n'.format(epoch + 1, args.epochs),
                  'Training Loss:{:.4f}\t Training Accuracy:{:.4f}\n'.format(train_loss_ave,
                                                                             train_accuracy_i / len(train_Loader)),
                  'Testing Loss:{:.4f}\t Testing Accuracy:{:.4f}\n'.format(val_loss_ave,
                                                                           test_accuracy_i / len(test_Loader)),
                  'Training MIoU:{:.4f}\t Testing MIoU:{:.4f}\n'.format(train_miou_i / len(train_Loader),
                                                                        test_miou_i / len(test_Loader)),
                  'Time: {:.4f}min'.format((time.time() - since_time) / 60))

    # plot the loss
    plt.figure(figsize=(25, 25))
    plt.subplot(221)
    plt.plot(train_accuracy, '-o', label="train_acc")
    plt.plot(test_accuracy, '-o', label="val_acc")
    plt.title("epoch_acc")
    plt.subplot(222)
    plt.plot(train_epoch_loss, '-o', label="train_loss")
    plt.plot(val_epoch_loss, '-o', label="val_loss")
    plt.title("epoch_loss")
    plt.subplot(223)
    plt.plot(train_miou, '-o', label="train_mIoU")
    plt.plot(test_miou, '-o', label="val_mIoU")
    plt.title("epoch_mIoU")
    plt.legend()
    plt.show()

    return train_accuracy, test_accuracy, train_epoch_loss, val_epoch_loss, train_miou, test_miou

def train_den(net, train_Loader, test_Loader, args):
    model = net.cuda()
    print("Model device:", next(model.parameters()).device)

    Loss = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,150], gamma=0.1)

    train_epoch_loss = []
    val_epoch_loss = []
    min_ssim = 0
    train_psnr = []
    test_psnr = []
    train_ssim = []
    test_ssim = []

    # Restore training state
    if args.resume:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Start epoch:", epoch + 1)

    # start train
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_ssim_i = 0
        train_psnr_i = 0
        since_time = time.time()
        for image, label in train_Loader:
            # for batch_idx,() in enumerate(train_Loader):
            image, label = trans(image), trans(label)
            image, label = image.cuda(), label.cuda()
            # print("Image device:", image.device, "Label device:", label.device)

            optimizer.zero_grad()  # Clear gradients before using
            output = model(image)
            # print(output.shape, label.shape)
            loss = Loss(output, label)   # Backpropagate loss to compute gradients for all tensors in the model
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # print(output.max(), output.min(), label.max(), label.min())
            psnr_i = 0
            ssim_i = 0
            label_np = label.squeeze(1).cpu().numpy()   # shape: (B, H, W)
            output_np = output.squeeze(1).detach().cpu().numpy() # shape: (B, H, W)
            # print(output_np.max(),output_np.min())

            for i in range(output_np.shape[0]):
                psnr_i += peak_signal_noise_ratio(label_np[i], output_np[i], data_range=1.0)
                ssim_i += structural_similarity(label_np[i], output_np[i], data_range=1.0)
            train_psnr_i += psnr_i / output_np.shape[0]
            train_ssim_i += ssim_i / output_np.shape[0]

        # Evaluation mode: no gradient backpropagation needed
        # Test accuracy on the validation dataset
        model.eval()
        val_loss = []
        test_psnr_i = 0
        test_ssim_i = 0
        with torch.no_grad():
            for image, label in test_Loader:
                image, label = trans(image), trans(label)
                image, label = image.cuda(), label.cuda()
                output = model(image)
                loss = Loss(output, label)
                val_loss.append(loss.item())

                # Save evaluation metrics or parameters
                psnr_i = 0
                ssim_i = 0
                label_np = label.squeeze(1).cpu().numpy()  # shape: (B, H, W)
                output_np = output.squeeze(1).detach().cpu().numpy()  # shape: (B, H, W)
                for i in range(output_np.shape[0]):
                    psnr_i += peak_signal_noise_ratio(label_np[i], output_np[i], data_range=1.0)
                    ssim_i += structural_similarity(label_np[i], output_np[i], data_range=1.0)

                test_psnr_i += psnr_i / output_np.shape[0]
                test_ssim_i += ssim_i / output_np.shape[0]

        # Save model parameters
        if test_ssim_i / len(test_Loader) > min_ssim:
            # Save the best-performing model
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                args.beat_path
            )
            print("best ssim has saved:{:.4f} --- > {:.4f}".format(min_ssim,
                                                                       (test_ssim_i / len(test_Loader))))
            min_ssim = test_ssim_i / len(test_Loader)

        #The last epoch save
        if epoch + 1 == args.epochs:
            # save model
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                args.save_path
            )
            print("train over")

        train_psnr.append(train_psnr_i / len(train_Loader))
        test_psnr.append(test_psnr_i / len(test_Loader))
        train_ssim.append(train_ssim_i / len(train_Loader))
        test_ssim.append(test_ssim_i / len(test_Loader))

        train_loss = torch.tensor(train_loss, device='cpu')
        train_loss_ave = np.average(train_loss)
        train_epoch_loss.append(train_loss_ave)

        val_loss = torch.tensor(val_loss, device='cpu')
        val_loss_ave = np.average(val_loss)
        val_epoch_loss.append(val_loss_ave)

        # Learning rate optimization
        scheduler.step()
        print('Learning rate: {:.6f}'.format(scheduler.get_last_lr()[0]))

        if (epoch) % 1 == 0:
            print('-------------------Epoch:{}/{}\t ------------------\n'.format(epoch + 1, args.epochs),
                  'Training Loss:{:.4f}\t Training PSNR:{:.4f}\n'.format(train_loss_ave,
                                                                             train_psnr_i / len(train_Loader)),
                  'Testing Loss:{:.4f}\t Testing PSNR:{:.4f}\n'.format(val_loss_ave,
                                                                           test_psnr_i / len(test_Loader)),
                  'Training SSIM:{:.4f}\t Testing SSIM:{:.4f}\n'.format(train_ssim_i / len(train_Loader),
                                                                        test_ssim_i / len(test_Loader)),
                  'Time: {:.4f}min'.format((time.time() - since_time) / 60))

    # plot the loss
    plt.figure(figsize=(25, 25))
    plt.subplot(221)
    plt.plot(train_psnr, '-o', label="train_acc")
    plt.plot(test_psnr, '-o', label="val_acc")
    plt.title("epoch_acc")
    plt.subplot(222)
    plt.plot(train_epoch_loss, '-o', label="train_loss")
    plt.plot(val_epoch_loss, '-o', label="val_loss")
    plt.title("epoch_loss")
    plt.subplot(223)
    plt.plot(train_ssim, '-o', label="train_mIoU")
    plt.plot(test_ssim, '-o', label="val_mIoU")
    plt.title("epoch_mIoU")
    plt.legend()
    plt.show()

    return train_psnr, test_psnr, train_epoch_loss, val_epoch_loss, train_ssim, test_ssim

def save_lists_to_excel(file_path, train_1, test_1,train_loss,test_loss,train_2,test_2, segment = True):
    if segment:
        list_dict = {
            'train_accuracy': train_1,
            'test_accuracy': test_1,
            'train_epoch_loss': train_loss,
            'val_epoch_loss': test_loss,
            'train_miou': train_2,
            'test_miou': test_2
        }
    else:
        list_dict = {
            'train_psnr': train_1,
            'test_psnr': test_1,
            'train_epoch_loss': train_loss,
            'val_epoch_loss': test_loss,
            'train_ssim': train_2,
            'test_ssim': test_2
        }
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, data_list in list_dict.items():
            df = pd.DataFrame(data_list)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Successfully save Excel file")

