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


syn_data_loader = load_syn2real_data(train_data_path, train_label_path, shuffle = True, batch_size = batch_size)

#tmp = next(iter(syn_data_loader)) #.dtype(torch.FloatTensor)
#print(tmp[0].shape)
#utils.save_image(tmp[0])

# Models
feature_net = BaseSimpleFeatureNet().type(dtype)
classify_net = ClassifierNet().type(dtype)
gen_net = BaseSimpleFeatureNet().type(dtype)
dis_net = BasicSimpleDiscNet().type(dtype)

if load_feature_net:
    feature_net.load_state_dict(torch.load('saved_models/feature_net.pkl'))

if load_classify_net:
    classify_net.load_state_dict(torch.load('saved_models/classify_net.pkl'))

# Train on synthetin data
feature_net_train(feature_net, classify_net, syn_data_loader, num_epochs = num_epochs, batch_size = batch_size)

gan_train_domain_adapt(gen_net, dis_net, real_data_loader, syn_data_loader, num_epochs = num_epochs, batch_size = batch_size)



