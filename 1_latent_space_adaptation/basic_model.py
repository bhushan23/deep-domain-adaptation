import torch
import sys
sys.path.insert(0, '../common')
sys.path.insert(0, './')
from torch.autograd import Variable
from data_loading import *
from models import *
from train import *


train_data_path = '../data/openset_classification/train'
train_label_path = train_data_path+'/image_list.txt'
batch_size = 64
num_epochs = 101

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!


data_loader = load_syn2real_data(train_data_path, train_label_path, shuffle = True, batch_size = batch_size)

#tmp = next(iter(data_loader)) #.dtype(torch.FloatTensor)
#print(tmp[0].shape)
#utils.save_image(tmp[0])

# Models
feature_net = BaseSimpleFeatureNet().type(dtype)
classify_net = ClassifierNet().type(dtype)

# Train
feature_net_train(feature_net, classify_net, data_loader, num_epochs = num_epochs, batch_size = batch_size)

