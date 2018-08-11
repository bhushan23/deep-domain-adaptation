import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../common')
from utils import *

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

def test(g_net, c_net, train_data, batch_size = 64, output_file='./test_results.txt'):
    prediction = torch.tensor([])
    for data in train_data:
        s = data
        mini_batch_size = s.shape[0]
        s = var(s).type(dtype)
        output = g_net(s)
        output = c_net(output)
        prediction = torch.cat((prediction, output))
    print(prediction.shape)
    return prediction


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

        print('Epoch:', epoch, 'Total Loss:', total_loss.cpu().item(), 'Last batch Loss:', loss.cpu().item())
        if epoch % 10 == 0:
            # torch.save(f_net.state_dict(), output_path+'saved_models/f_net_'+str(epoch/100)+'.pkl')
            # torch.save(c_net.state_dict(), output_path+'saved_models/l_net_'+str(epoch/100)+'.pkl')
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
            print('Epoch:', epoch, 'Accuracy:', accuracy.cpu().item())

def gan_train_domain_adapt(gen_net, dis_net, classify_net, real_data_loader, syn_data_loader, num_epochs = 10, batch_size = 64, lr = 0.001, output_path = './'):
    g_opt = torch.optim.Adam(gen_net.parameters(), lr = lr)
    d_opt = torch.optim.Adam(dis_net.parameters(), lr = lr)
    last_index = int(len(real_data_loader.dataset) / batch_size)
    for epoch in range(0, num_epochs):
        i = 0
        g_loss_total = 0
        d_loss_total = 0
        syn_img_iter = iter(syn_data_loader)
        syn_img_cnt = 0
        syn_img_len = len(syn_data_loader)
        for r_data in real_data_loader:
            # print(r_data[0].shape)
            real_img = r_data[0]
            real_label = r_data[1]
            i += 1
            if syn_img_cnt == syn_img_len:
                syn_img_iter = iter(syn_data_loader)
                syn_img_cnt = 0
            syn_img_cnt += 1
            syn_img = next(syn_img_iter)[0]
            mini_batch_size = syn_img.shape[0]
            syn_img = var(syn_img).type(dtype)
            real_img = var(real_img).type(dtype)

            if i == last_index:
                mini_batch_size = real_img.shape[0]
                real_image = var(real_img).type(dtype)
                real_label = var(torch.LongTensor(real_label)) #.type(dtype)
                output = gen_net(real_image)
                output = classify_net(output)
                _, predicted = torch.max(output.data, 1)

                accuracy = ((predicted == real_label).sum()) * 100 / mini_batch_size
                print('Epoch:', epoch, 'Accuracy:', accuracy.cpu().item())
                break

            # Discriminator training
            d_truth = dis_net(gen_net(syn_img))
            d_fake = dis_net(gen_net(real_img))
            d_loss = torch.mean(d_truth) - torch.mean(d_fake)

            dis_net.zero_grad()
            d_loss.sum().backward()
            d_opt.step()

            # Generator training
            g_fake = dis_net(gen_net(real_img))
            g_loss = -torch.mean(g_fake)
            gen_net.zero_grad()
            g_loss.backward()
            g_opt.step()
            d_loss_total += d_loss
            g_loss_total += g_loss

        #print('Epoch [{}/{}], Discriminator {}|{}, Generator {}|{}'.format(epoch+1, num_epochs, d_loss.item(), d_loss_total.item(), g_loss.item(), g_loss_total.item()))
        print('Epoch [{}/{}], Discriminator {}|{}, Generator {}|{}'.format(epoch+1, num_epochs, d_loss.cpu().item(), d_loss_total.cpu().item(), g_loss.cpu().item(), g_loss_total.cpu().item()))

        if epoch % 1== 0:
            torch.save(gen_net.state_dict(), output_path+'saved_models/g_net.pkl')
            torch.save(dis_net.state_dict(), output_path+'saved_models/d_net.pkl')
