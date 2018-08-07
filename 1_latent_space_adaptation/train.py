import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../common')
from utils import *

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

def feature_net_train(f_net, c_net, train_data, lr = 0.001, batch_size = 64, num_epochs = 5, output_path = './', validate = True):
    f_opt = torch.optim.Adam(f_net.parameters(), lr = lr)
    c_opt = torch.optim.Adam(c_net.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()
    
    last_index = int(len(train_data.dataset) / batch_size)
    for epoch in range(0, num_epochs):
        total_loss = 0
        i = 0
        for data in train_data:
            s = data[0]
            l = data[1]
            i += 1
            #if i == 10:
            #    break;
            if validate == True and i == last_index:
                break
            batchSize = s.shape[0]
            s = var(s).type(dtype)
            l = var(torch.LongTensor(l)) #.type(torch.LongTensor)
            output = f_net(s)
            output = c_net(output)
            loss = loss_fn(output, l)
            f_net.zero_grad()
            c_net.zero_grad()
            loss.backward()
            f_opt.step()
            c_opt.step()
            total_loss += loss
        
        print('Epoch:', epoch, 'Total Loss:', total_loss.cpu().data[0], 'Last batch Loss:', loss.cpu().data[0])
        if epoch+1 % 50 == 0:
            torch.save(f_net.state_dict(), output_path+'saved_models/f_net_'+str(epoch/100)+'.pkl')
            torch.save(c_net.state_dict(), output_path+'saved_models/l_net_'+str(epoch/100)+'.pkl')
            torch.save(f_net.state_dict(), output_path+'saved_models/f_net.pkl')
            torch.save(c_net.state_dict(), output_path+'saved_models/l_net.pkl')
        
        ## Validation accuracy on last index
        if validate:
            mini_batch_size = s.shape[0]
            s = var(s).type(dtype)
            l = var(torch.LongTensor(l)) #.type(dtype)
            output = f_net(s)
            output = c_net(output)
            _, predicted = torch.max(output.data, 1)
            # predicted = torch.LongTensor(predicted)
            accuracy = ((predicted == l).sum()) * 100 / mini_batch_size    
            print('Epoch:', epoch, 'Accuracy:', accuracy.cpu())



