from unet import Unet
from pkg import data_devide, data_read, data_load, Args, train, save_lists_to_excel
import os
import torch

os.environ['CUDA_VISIBLE_DIVICES']='0'

# define the model
net = Unet(drop=0)
# change the data path according to your dataset
images_path = '../../dataset/image/'
labels_path = '../../dataset/circularMask/'

# whether to devide the dataset into train and test
devide = False
#dataset prepare
if devide:
    data_devide(images_path, labels_path)
    print("sucessfully devide the image")

# read the dataset, you can set visual to True to visualize the dataset
images,labels,images_test,labels_test = data_read(visual = False, vis_num = 5)

# Important! Set the train parameter
args = Args(batch_size = 12, lr = 0.001, epochs = 200,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            save_path = 'output/unet_0.001.pth', beat_path = 'output/unet_best_0.001.pth',
            resume = False, resume_path = 'output/unet_best_0.001.pth')

train_Loader,test_Loader = data_load(args)

print('____train start_____')
train_accuracy,test_accuracy,train_epoch_loss,val_epoch_loss,train_miou,test_miou = train(net, train_Loader, test_Loader, args)

save_lists_to_excel('output/train_out.xlsx', train_accuracy,test_accuracy,
                    train_epoch_loss,val_epoch_loss,train_miou,test_miou )


