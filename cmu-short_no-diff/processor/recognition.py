import os
import sys
import argparse
import yaml
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .data_tools import *


# from ..net.model import Discriminator
from copy import deepcopy

# my_dict = {'one': 1, 'two': 2}
# new_dict_deepcopy = deepcopy(my_dict)
from torch.distributions.uniform import Uniform


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)
        V, W, U = 26, 10, 5
        off_diag_joint, off_diag_part, off_diag_body = np.ones([V, V])-np.eye(V, V), np.ones([W, W])-np.eye(W, W), np.ones([U, U])-np.eye(U, U)
        self.relrec_joint = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_joint)[1]), dtype=np.float32)).to(self.dev)
        self.relsend_joint = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_joint)[0]), dtype=np.float32)).to(self.dev)
        self.relrec_part = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_part)[1]), dtype=np.float32)).to(self.dev)
        self.relsend_part = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_part)[0]), dtype=np.float32)).to(self.dev)
        self.relrec_body = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_body)[1]), dtype=np.float32)).to(self.dev)
        self.relsend_body = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_body)[0]), dtype=np.float32)).to(self.dev)

        self.dismodel_args = deepcopy(self.arg.model_args)
        self.dismodel_args.pop('n_in_dec', None)
        self.dismodel_args.pop('n_hid_dec', None)
        self.dismodel_args.pop('n_hid_enc', None)
        self.dismodel_args['edge_weighting'] =True
        self.dismodel_args['fusion_layer'] = 0


        self.discriminator = self.io.load_model('net.model.Discriminator', **(self.dismodel_args))
        self.discriminator.apply(weights_init)
        self.discriminator.cuda()
        self.criterion = nn.BCEWithLogitsLoss()# nn.BCELoss()
        self.visual_sigmoid = nn.Sigmoid()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(params=self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)

            self.netD_optimizer =optim.Adam(params=self.discriminator.parameters(),
                                        lr=0.000001,
                                        weight_decay=self.arg.weight_decay)


    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (0.5**np.sum(self.meta_info['iter']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        elif self.arg.optimizer == 'Adam' and self.arg.step:
            lr = self.arg.base_lr * (0.98**np.sum(self.meta_info['iter']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            # # adjust lr
            # for param_group in self.netD_optimizer.param_groups:
            #     param_group['lr'] = lr
            self.lr = lr
        else:
            raise ValueError('No such Optimizer')


    def loss_l1(self, pred, target, mask=None):
        dist = torch.abs(pred-target).mean(-1).mean(1).mean(0)
        if mask is not None:
            dist = dist * mask
        loss = torch.mean(dist)
        return loss

    def vae_loss_function(self, x, x_hat, mean, log_var):
        # BCE_loss = nn.BCELoss()
        # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        assert x_hat.shape == x.shape
        # reconstruction_loss = torch.mean(torch.norm(x - x_hat, dim=len(x.shape) - 1))
        reconstruction_loss = self.loss_l1(x, x_hat)#  torch.mean(torch.norm(x - x_hat, dim=len(x.shape) - 1))
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstruction_loss + self.arg.kl_alpha * KLD, self.arg.kl_alpha *KLD


    def train(self):
        if self.meta_info['iter'] % 5 == 0:
            self.train_generator(mode='generator')
        else:
            with torch.no_grad():
                mean, var, gan_decoder_inputs, gan_targets = self.train_generator(mode='discriminator')


            self.train_decoder(mean, var, gan_decoder_inputs, gan_targets)


    def train_decoder(self, mean, var, gan_decoder_inputs, gan_targets):
        with torch.no_grad():
            dec_mean = mean.clone()
            dec_var = var.clone()
            dec_var = torch.exp(0.5 * dec_var) # TBD
            epsilon = torch.randn_like(dec_var)
            z = dec_mean + dec_var * epsilon
            dis_pred = self.model.generate_from_decoder(z, gan_decoder_inputs, self.arg.target_seq_len) #[32, 26, 10, 3]

            dis_pred = dis_pred.permute(0, 2, 1, 3).contiguous().view(32, 10, -1)
            dis_o = self.discriminator(dis_pred)# .view(-1)

            dis_o = dis_o.detach()
            dis_o =dis_o.requires_grad_()



        self.netD_optimizer.zero_grad()
        N = dis_o.size()[0]
        # label = torch.full((N,), 0.0, dtype=torch.float, device='cuda:0')
        # label = Uniform(0.0, 0.1).sample((N,1)).cuda()
        fake_labels = torch.FloatTensor(1).fill_(0.0)
        fake_labels = fake_labels.requires_grad_(False)
        fake_labels = fake_labels.expand_as(dis_o).cuda()
        # print(fake_labels.size())
        # print(dis_o.size())
        errD_fake= self.criterion(dis_o, fake_labels)
        # Calculate gradients for D in backward pass
        errD_fake.backward()
        D_x_fake = dis_o.mean().item() # to display

        # for the real
        targets = gan_targets#.permute(0, 2, 1, 3).contiguous().view(32, 10, -1)
        dis_oreal = self.discriminator(targets)# .view(-1)
        # real_labels = torch.full((N,), 1.0, dtype=torch.float, device='cuda:0')
        # real_labels = Uniform(0.9, 1.0).sample((N,1)).cuda()
        real_labels = torch.FloatTensor(1).fill_(1.0)
        real_labels = real_labels.requires_grad_(False)
        real_labels  = real_labels.expand_as(dis_oreal).cuda()
        # print(real_labels.requires_grad)
        errD_real= self.criterion(dis_oreal, real_labels)
        errD_real.backward()
        D_x_real = dis_oreal.mean().item()
        errD = errD_real + errD_fake

        self.netD_optimizer.step()
        self.iter_info['discriminator loss'] = errD
        self.iter_info['discriminator real out'] = D_x_real
        self.iter_info['discriminator fake out'] = D_x_fake
        self.iter_info['discriminator real loss'] = errD_real
        self.iter_info['discriminator fake loss'] = errD_fake

        self.show_iter_info()
        self.meta_info['iter'] += 1

        # self.epoch_info['mean_loss']= np.mean(loss_value)

        # print(dis_o.size())

    def train_generator(self, mode='generator'):
        self.model.train()
        self.adjust_lr()
        loss_value = []
        normed_train_dict = normalize_data(self.train_dict, self.data_mean, self.data_std, self.dim_use)

        encoder_inputs, decoder_inputs, targets = train_sample(normed_train_dict,
                                                               self.arg.batch_size,
                                                               self.arg.source_seq_len,
                                                               self.arg.target_seq_len,
                                                               len(self.dim_use))
        encoder_inputs_v = np.zeros_like(encoder_inputs)
        encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
        encoder_inputs_a = np.zeros_like(encoder_inputs)
        encoder_inputs_a[:, :-1, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

        encoder_inputs_p = torch.Tensor(encoder_inputs).float().to(self.dev)
        encoder_inputs_v = torch.Tensor(encoder_inputs_v).float().to(self.dev)
        encoder_inputs_a = torch.Tensor(encoder_inputs_a).float().to(self.dev)

        decoder_inputs = torch.Tensor(decoder_inputs).float().to(self.dev)
        decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1).to(self.dev)
        decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1).to(self.dev)
        targets = torch.Tensor(targets).float().to(self.dev)                            # [N,T,D] = [64, 10, 63]
        gan_targets = targets.clone().detach().requires_grad_(True)
        N, T, D = targets.size()                                                        # N = 64(batchsize), T=10, D=63
        targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)          # [64, 21, 10, 3]

        gan_decoder_inputs = decoder_inputs.clone().detach().requires_grad_(True)


        outputs_combo = self.model(encoder_inputs_p,
                             encoder_inputs_v,
                             encoder_inputs_a,
                             decoder_inputs,
                             decoder_inputs_previous,
                             decoder_inputs_previous2,
                             self.arg.target_seq_len,
                             self.relrec_joint,
                             self.relsend_joint,
                             self.relrec_part,
                             self.relsend_part,
                             self.relrec_body,
                             self.relsend_body,
                             self.arg.lamda)

        outputs, mean, var = outputs_combo
        # loss_l1 = self.loss_l1(outputs, targets)

        if mode == 'generator':
            kl_loss, kl = self.vae_loss_function(outputs, targets, mean, var)
            loss = kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['kl_loss'] = kl.data.item()
            # self.iter_info['loss_l1'] = loss_l1.data.item()

            self.show_iter_info()
            self.meta_info['iter'] += 1

            self.epoch_info['mean_loss']= np.mean(loss_value)

        return mean, var, gan_decoder_inputs, gan_targets




    def test(self, evaluation=True, iter_time=0, save_motion=False, phase=False):

        self.model.eval()
        loss_value = []
        normed_test_dict = normalize_data(self.test_dict, self.data_mean, self.data_std, self.dim_use)
        self.actions = ["basketball", "basketball_signal", "directing_traffic", 
                   "jumping", "running", "soccer", "walking", "washwindow"]

        self.io.print_log(' ')
        print_str = "{0: <16} |".format("milliseconds")
        for ms in [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 560, 1000]:
            print_str = print_str + " {0:5d} |".format(ms)
        self.io.print_log(print_str)

        for action_num, action in enumerate(self.actions):
            encoder_inputs, decoder_inputs, targets = srnn_sample(normed_test_dict, action,
                                                                  self.arg.source_seq_len, 
                                                                  self.arg.target_seq_len, 
                                                                  len(self.dim_use))
            encoder_inputs_v = np.zeros_like(encoder_inputs)
            encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
            encoder_inputs_a = np.zeros_like(encoder_inputs)
            encoder_inputs_a[:, 1:, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

            encoder_inputs_p = torch.Tensor(encoder_inputs).float().to(self.dev)                         # [N,T,D] = [64, 49, 63]
            encoder_inputs_v = torch.Tensor(encoder_inputs_v).float().to(self.dev)                       # [N,T,D] = [64, 49, 63]
            encoder_inputs_a = torch.Tensor(encoder_inputs_a).float().to(self.dev)

            decoder_inputs = torch.Tensor(decoder_inputs).float().to(self.dev)                           # [N,T,D] = [64,  1, 63]
            decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1).to(self.dev)   # [N,T,D] = [64,  1, 63]
            decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1).to(self.dev)
            targets = torch.Tensor(targets).float().to(self.dev)                                         # [N,T,D] = [64, 25, 63]
            N, T, D = targets.size()                                                         
            targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)                         # [64, 21, 25, 3]  same with outputs for validation loss

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(encoder_inputs_p,
                                     encoder_inputs_v,
                                     encoder_inputs_a,
                                     decoder_inputs,
                                     decoder_inputs_previous,
                                     decoder_inputs_previous2,
                                     self.arg.target_seq_len,
                                     self.relrec_joint,
                                     self.relsend_joint,
                                     self.relrec_part,
                                     self.relsend_part,
                                     self.relrec_body,
                                     self.relsend_body,
                                     self.arg.lamda)

            if evaluation:
                mean_errors = np.zeros((8, 25), dtype=np.float32)
                for i in np.arange(8):

                    output = outputs[0][i]                   # output: [V, t, d] = [21, 25, 3]
                    V, t, d = output.shape
                    output = output.permute(1,0,2).contiguous().view(t, V*d)
                    output_denorm = unnormalize_data(output.cpu().numpy(), self.data_mean, self.data_std, self.dim_ignore, self.dim_use, self.dim_zero)
                    t, D = output_denorm.shape
                    output_euler = np.zeros((t,D) , dtype=np.float32)        # [21, 99]
                    for j in np.arange(t):
                        for k in np.arange(0,115,3):
                            output_euler[j,k:k+3] = rotmat2euler(expmap2rotmat(output_denorm[j,k:k+3]))

                    target = targets[i]
                    target = target.permute(1,0,2).contiguous().view(t, V*d)
                    target_denorm = unnormalize_data(target.cpu().numpy(), self.data_mean, self.data_std, self.dim_ignore, self.dim_use, self.dim_zero)
                    target_euler = np.zeros((t,D) , dtype=np.float32)
                    for j in np.arange(t):
                        for k in np.arange(0,115,3):
                            target_euler[j,k:k+3] = rotmat2euler(expmap2rotmat(target_denorm[j,k:k+3]))

                    target_euler[:,0:6] = 0
                    idx_to_use1 = np.where(np.std(target_euler,0)>1e-4)[0]
                    idx_to_use2 = self.dim_nonzero
                    idx_to_use =  idx_to_use1[np.in1d(idx_to_use1,idx_to_use2)]
                    
                    euc_error = np.power(target_euler[:,idx_to_use]-output_euler[:,idx_to_use], 2)
                    euc_error = np.sqrt(np.sum(euc_error, 1))    # [25]
                    mean_errors[i,:euc_error.shape[0]] = euc_error
                mean_mean_errors = np.mean(np.array(mean_errors), 0)

                if save_motion==True:
                    save_dir = os.path.join(self.save_dir,'motions_exp'+str(iter_time*self.arg.savemotion_interval))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    np.save(save_dir+'/motions_'+action+'.npy', outputs.cpu().numpy())

                print_str = "{0: <16} |".format(action)
                for ms_idx, ms in enumerate([0,1,2,3,4,5,6,7,8,9,13,24]):
                    if self.arg.target_seq_len >= ms+1:
                        print_str = print_str + " {0:.3f} |".format(mean_mean_errors[ms])
                        if phase is not True:
                            self.MAE_tensor[iter_time, action_num, ms_idx] = mean_mean_errors[ms]
                    else:
                        print_str = print_str + "   n/a |"
                        if phase is not True:
                            self.MAE_tensor[iter_time, action_num, ms_idx] = 0
                print_str = print_str + 'T: {0:.3f} ms |'.format((time.time()-start_time)*1000/8)
                self.io.print_log(print_str)
        self.io.print_log(' ')


    @staticmethod
    def get_parser(add_help=False):

        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help, parents=[parent_parser], description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

        parser.add_argument('--lamda', type=float, default=1.0, help='adjust part feature')
        parser.add_argument('--fusion_layer_dir', type=str, default='fusion_1', help='lamda a dir')
        parser.add_argument('--learning_rate_dir', type=str, default='adam_1e-4', help='lamda a dir')
        parser.add_argument('--lamda_dir', type=str, default='nothing', help='adjust part feature')
        parser.add_argument('--crossw_dir', type=str, default='nothing', help='adjust part feature')
        parser.add_argument('--note', type=str, default='nothing', help='whether seperate')
        parser.add_argument('--kl_alpha', type=float, default=0.1)

        parser.add_argument('--debug', type=bool, default=False, help='whether seperate')

        return parser
