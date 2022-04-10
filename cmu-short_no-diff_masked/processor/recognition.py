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
        self.lower_body_joints = [1, 2, 3]
        
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
            self.lr = lr
        else:
            raise ValueError('No such Optimizer')

    def loss_l1(self, pred, target, mask=None):
        dist = torch.abs(pred-target).mean(-1).mean(1).mean(0)
        if mask is not None:
            dist = dist * mask
        loss = torch.mean(dist)
        return loss

    def vae_loss_function(self, pred, target, mean_val, log_var):
        assert pred.shape == target.shape
        reconstruction_loss = self.loss_l1(pred, target)
        mean_val = mean_val.mean(-1).mean(1).mean(0)
        log_var = log_var.mean(-1).mean(1).mean(0)
        KLD = - 0.5 * torch.sum(1+ log_var - mean_val.pow(2) - log_var.exp())
        return reconstruction_loss + 0.1*KLD    


    def build_masking_matrix_add_noise(self, unmasked_matrix, joint_indices):
        r"""
        Build masking matrix with same shape as `unmasked_matrix`
        """
        M = np.zeros_like(unmasked_matrix)
        M = M.reshape(M.shape[0], M.shape[1], -1, 3) # batch size, T, J, 3
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                for k in range(M.shape[2]):
                    if k in joint_indices:
                        M[i, j, k, :] = np.random.normal(0,0.5,1)       
        #M[:, :, joint_indices, :] = np.random.normal(0,0.5,3)
        M = M.reshape(unmasked_matrix.shape)
        return M      

    def build_masking_matrix(self, unmasked_matrix, joint_indices):
        r"""
        Build masking matrix with same shape as `unmasked_matrix`
        """
        M = np.ones_like(unmasked_matrix)
        M = M.reshape(M.shape[0], M.shape[1], -1, 3) # batch size, T, J, 3
        M[:, :, joint_indices, :] = np.zeros((3,))
        M = M.reshape(unmasked_matrix.shape)
        return M
    
    def build_lower_body_masking_matrices(self, lower_body_joints, encoder_inputs, decoder_inputs):
        # build encoder input mask
        M_enc_in = self.build_masking_matrix(encoder_inputs, lower_body_joints)
        # build decoder input mask
        M_dec_in = self.build_masking_matrix(decoder_inputs, lower_body_joints)
        # build decoder output / target mask
        #M_dec_out = self.build_masking_matrix(targets, lower_body_joints)
        return M_enc_in, M_dec_in
    
    def train(self):
        self.model.train()
        self.adjust_lr()
        loss_value = []
        normed_train_dict = normalize_data(self.train_dict, self.data_mean, self.data_std, self.dim_use)

        encoder_inputs, decoder_inputs, targets = train_sample(normed_train_dict, 
                                                               self.arg.batch_size, 
                                                               self.arg.source_seq_len, 
                                                               self.arg.target_seq_len, 
                                                               len(self.dim_use))

        #build lower-body masking matrices
        self.M_enc_in, self.M_dec_in = self.build_lower_body_masking_matrices(
            self.lower_body_joints,
            encoder_inputs,
            decoder_inputs
        )

        decoder_noise = self.build_masking_matrix_add_noise(decoder_inputs, self.lower_body_joints)
        encoder_noise = self.build_masking_matrix_add_noise(encoder_inputs, self.lower_body_joints)
        
        #mask encoder inputs and decoder inputs
        encoder_inputs = np.multiply(self.M_enc_in, encoder_inputs)
        decoder_inputs = np.multiply(self.M_dec_in, decoder_inputs)
        decoder_inputs = np.add(decoder_inputs,decoder_noise)

        encoder_inputs_with_noise = encoder_inputs.copy()
        encoder_inputs_with_noise = np.add(encoder_inputs_with_noise,encoder_noise)

        encoder_inputs_v = np.zeros_like(encoder_inputs)
        encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
        encoder_inputs_a = np.zeros_like(encoder_inputs)
        encoder_inputs_a[:, :-1, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

        encoder_inputs_p = torch.Tensor(encoder_inputs).float().to(self.dev)
        encoder_inputs_v = torch.Tensor(encoder_inputs_v).float().to(self.dev)
        encoder_inputs_a = torch.Tensor(encoder_inputs_a).float().to(self.dev)

        decoder_inputs = torch.Tensor(decoder_inputs).float().to(self.dev)
        #decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1).to(self.dev)
        #decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1).to(self.dev)
        decoder_inputs_previous = torch.Tensor(encoder_inputs_with_noise[:, -1, :]).unsqueeze(1).to(self.dev)
        decoder_inputs_previous2 = torch.Tensor(encoder_inputs_with_noise[:, -2, :]).unsqueeze(1).to(self.dev)
        targets = torch.Tensor(targets).float().to(self.dev)                            # [N,T,D] = [64, 10, 63]
        N, T, D = targets.size()                                                        # N = 64(batchsize), T=10, D=63
        targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)          # [64, 21, 10, 3]

        outputs, mean, log_var = self.model(encoder_inputs_p,
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

        # convert spatio-temporal masking matrix to a tensor
        #st_mask = torch.from_numpy(self.M_dec_out).to(self.dev)
        #loss = self.vae_loss_function(outputs, targets, mean, log_var, st_mask = st_mask)

        loss = self.vae_loss_function(outputs, targets, mean, log_var)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.iter_info['loss'] = loss.data.item()
        self.show_iter_info()
        self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)


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

            #build lower-body masking matrices
            self.M_enc_in, self.M_dec_in = self.build_lower_body_masking_matrices(
                self.lower_body_joints,
                encoder_inputs,
                decoder_inputs
            )
            
            decoder_noise = self.build_masking_matrix_add_noise(decoder_inputs, self.lower_body_joints)
            encoder_noise = self.build_masking_matrix_add_noise(encoder_inputs, self.lower_body_joints)
        
            #mask encoder inputs and decoder inputs
            encoder_inputs = np.multiply(self.M_enc_in, encoder_inputs)
            decoder_inputs = np.multiply(self.M_dec_in, decoder_inputs)
            decoder_inputs = np.add(decoder_inputs,decoder_noise)

            encoder_inputs_with_noise = encoder_inputs.copy()
            encoder_inputs_with_noise = np.add(encoder_inputs_with_noise,encoder_noise)

            encoder_inputs_v = np.zeros_like(encoder_inputs)
            encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
            encoder_inputs_a = np.zeros_like(encoder_inputs)
            encoder_inputs_a[:, 1:, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

            encoder_inputs_p = torch.Tensor(encoder_inputs).float().to(self.dev)                         # [N,T,D] = [64, 49, 63]
            encoder_inputs_v = torch.Tensor(encoder_inputs_v).float().to(self.dev)                       # [N,T,D] = [64, 49, 63]
            encoder_inputs_a = torch.Tensor(encoder_inputs_a).float().to(self.dev)
            # for saving motion
            N, T, D = encoder_inputs_p.shape
            encoder_inputs_p_4d = encoder_inputs_p.view(N, T, -1, 3).permute(0, 2, 1, 3)                 # Eric: [N, V, T, 3]  same with targets for saving motion

            decoder_inputs = torch.Tensor(decoder_inputs).float().to(self.dev)                           # [N,T,D] = [64,  1, 63]
            #decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1).to(self.dev)
            #decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1).to(self.dev)
            decoder_inputs_previous = torch.Tensor(encoder_inputs_with_noise[:, -1, :]).unsqueeze(1).to(self.dev)
            decoder_inputs_previous2 = torch.Tensor(encoder_inputs_with_noise[:, -2, :]).unsqueeze(1).to(self.dev)
            targets = torch.Tensor(targets).float().to(self.dev)                                         # [N,T,D] = [64, 25, 63]
            N, T, D = targets.size()                                                         
            targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)                         # [64, 21, 25, 3]  same with outputs for validation loss

            start_time = time.time()
            with torch.no_grad():
                outputs, mean, var = self.model(encoder_inputs_p,
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
                num_samples_per_action = encoder_inputs_p_4d.shape[0]
                mean_errors = np.zeros(
                    (num_samples_per_action, self.arg.target_seq_len), dtype=np.float32)
                # Eric: create data structs to save unnormalized inputs, outputs and targets
                inputs_denorm = np.zeros(
                    [num_samples_per_action,
                    encoder_inputs_p_4d.shape[2],
                    int(self.data_mean.shape[0]/3),
                    3]) # num_samples_per_action, t_in, 39, 3
                outputs_denorm = np.zeros(
                    [num_samples_per_action,
                    outputs.shape[2],
                    int(self.data_mean.shape[0]/3),
                    3]) # [num_samples_per_action, t_out, 39, 3]
                targets_denorm = np.zeros(
                    [num_samples_per_action,
                    targets.shape[2],
                    int(self.data_mean.shape[0]/3),
                    3]) # [num_samples_per_action, t_out, V, 3]
                
                for i in np.arange(num_samples_per_action):
                    input = encoder_inputs_p_4d[i] # V, t_in, d
                    V, t, d = input.shape
                    input = input.permute(1,0,2).contiguous().view(t, V*d)
                    input_denorm = unnormalize_data(
                        input.cpu().numpy(), self.data_mean, self.data_std, self.dim_ignore, self.dim_use, self.dim_zero)
                    inputs_denorm[i] = input_denorm.reshape((t, -1, 3))
                    
                    output = outputs[i]                   # output: [V, t, d] = [21, 25, 3]
                    V, t, d = output.shape
                    output = output.permute(1,0,2).contiguous().view(t, V*d)
                    output_denorm = unnormalize_data(
                        output.cpu().numpy(), self.data_mean, self.data_std, self.dim_ignore, self.dim_use, self.dim_zero)
                    outputs_denorm[i] = output_denorm.reshape((t, -1, 3))
                    t, D = output_denorm.shape
                    output_euler = np.zeros((t,D) , dtype=np.float32)        # [21, 99]
                    for j in np.arange(t):
                        for k in np.arange(0,115,3):
                            output_euler[j,k:k+3] = rotmat2euler(expmap2rotmat(output_denorm[j,k:k+3]))

                    target = targets[i]
                    target = target.permute(1,0,2).contiguous().view(t, V*d)
                    target_denorm = unnormalize_data(
                        target.cpu().numpy(), self.data_mean, self.data_std, self.dim_ignore, self.dim_use, self.dim_zero)
                    targets_denorm[i] = target_denorm.reshape((t, -1, 3))
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
                    # save unnormalized inputs
                    np.save(save_dir+f"/motions_{action}_inputs.npy", inputs_denorm)
                    # save unnormalized outputs
                    np.save(save_dir+f"/motions_{action}_outputs.npy", outputs_denorm)
                    # save unnormalized targets
                    np.save(save_dir+f"/motions_{action}_targets.npy", targets_denorm)

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

        parser.add_argument('--debug', type=bool, default=False, help='whether seperate')

        return parser
