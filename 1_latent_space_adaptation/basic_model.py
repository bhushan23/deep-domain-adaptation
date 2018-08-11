import torch
import sys
sys.path.insert(0, '../common')
sys.path.insert(0, './')
from torch.autograd import Variable
from data_loading import *
from models import *
from train import *

syn_data_path = '../data/openset_classification/train/'
syn_label_path = syn_data_path + 'image_list.txt'

real_data_path = '../data/openset_classification/validation/'
real_label_path = real_data_path + 'image_list.txt'

test_data_path = '../data/openset_classification/test/'

batch_size = 64
num_epochs = 5

load_feature_net = True
load_classify_net = True
test = True

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

 # Models
feature_net = BaseSimpleFeatureNet().type(dtype)
classify_net = ClassifierNet().type(dtype)
gen_net = BaseSimpleFeatureNet().type(dtype)
dis_net = Discriminator().type(dtype)

classify_net.load_state_dict(torch.load('saved_models/l_net.pkl'))
gen_net.load_state_dict(torch.load('saved_models/g_net.pkl'))
c_cpu = classify.cpu()
g_cpu = gen_net.cpu()

torch.save(c_cpu.state_dict(), output_path+'saved_models/c_cpu__net.pkl')
torch.save(g_cpu.state_dict(), output_path+'saved_models/g_cpu_net.pkl')

if test == True:
    test_data_loader = load_syn2real_data(test_data_path, shuffle = False, batch_size = batch_size)
    classify_net.load_state_dict(torch.load('saved_models/l_net.pkl'))
    gen_net.load_state_dict(torch.load('saved_models/g_net.pkl'))
    test(gen_net, classify_net, test_data_loader)
else:
    syn_data_loader = load_syn2real_data(syn_data_path, syn_label_path, shuffle = True, batch_size = batch_size)
    real_data_loader = load_syn2real_data(real_data_path, real_label_path, shuffle = True, batch_size = batch_size)

    tmp = next(iter(real_data_loader)) #.dtype(torch.FloatTensor)
    print(tmp[0].shape)
    #utils.save_image(tmp[0])

    if load_feature_net:
        feature_net.load_state_dict(torch.load('saved_models/f_net.pkl'))

    if load_classify_net:
        classify_net.load_state_dict(torch.load('saved_models/l_net.pkl'))

    # Train on synthetin data
    #feature_net_train(feature_net, classify_net, syn_data_loader, num_epochs = 1, batch_size = batch_size)
    #feature_net_train(feature_net, classify_net, syn_data_loader, num_epochs = 1, batch_size = batch_size, lr = 0.0001)

    gan_train_domain_adapt(gen_net, dis_net, classify_net, real_data_loader, syn_data_loader, num_epochs = num_epochs, batch_size = batch_size)
    gan_train_domain_adapt(gen_net, dis_net, classify_net, real_data_loader, syn_data_loader, num_epochs = num_epochs, batch_size = batch_size, lr = 0.0001)
