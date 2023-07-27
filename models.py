
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RNNImputation(nn.Module):
    def __init__(self,input_size=1, hidden_size=128):
        super(RNNImputation, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRUCell(input_size,hidden_size)
        self.fc = nn.Linear(hidden_size,input_size);

    def forward(self, input, device):
        batch_size = input.shape[0] #input[bs,seq_len.input_size]
        input = input.permute(1, 0, 2) #[seq_len, bs, input_size]
        seq_steps = input.shape[0]
        h = torch.zeros([batch_size, self.hidden_size], dtype=torch.float32)
        outputs = []
        new_input = []
        for step in range(seq_steps):
            if step == 0:
                h = self.rnn(input[step])
                outputs.append(h)
                new_input.append(input[step])
            else:
                comparison = input[step] == torch.tensor(128, dtype=torch.float32)
                x_hat = self.fc(outputs[step - 1])
                current_input = torch.where(comparison, x_hat, input[step])
                h = self.rnn(current_input, h)
                outputs.append(h)
                new_input.append(current_input)

        prediction_target_hidden_output = outputs[:-1] #seq_len - 1 [bs, hidden_size]
        prediction_hidden = torch.reshape(input=torch.cat(prediction_target_hidden_output, dim=1), shape=[-1, self.hidden_size]) # [seq_len-1 * bs, hidden_size]
        prediction = self.fc(prediction_hidden) #[seq_len-1 * bs, input_size]
        new_input = torch.tensor([item.cpu().detach().numpy() for item in new_input]).to(device) #.cuda(1) [seq_len, bs, input_size]

        return new_input, prediction


class SampaddingConv1D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size - 1) / 2), int(kernel_size / 2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      dilation=4,
                                      padding='same')
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        # x = self.padding(x)
        x = self.conv1d(x)
        x = self.bn(x)
        return x


class build_layer_with_layer_parameter(nn.Module):
    """
    formerly build_layer_with_layer_parameter
    """
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            # in_channels, out_channels, kernel_size
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, x):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(x)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result



def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = max(1, int(paramenter_layer / (in_channel * sum(prime_list))))
    return out_channel_expect


def generate_layer_parameter_list(start, end, layers, in_channel=1):

    prime_list = [7, 11, 15, 19, 23, 27]

    layer_parameter_list = []
    for paramenter_number_of_layer in layers:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)

        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list) * out_channel

        layer_parameter_list.append(tuples_in_layer)

    tuples_in_layer_last = []
    first_out_channel = len(prime_list) * get_out_channel_number(layers[0], 1, prime_list)
    tuples_in_layer_last.append((in_channel, first_out_channel, 1))
    tuples_in_layer_last.append((in_channel, first_out_channel, 3))
    tuples_in_layer_last.append((in_channel, first_out_channel, 5))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


class Model(nn.Module):
    def __init__(self, in_features=1,
                 out=1, kernel_size=1,
                 seq_len=300, dmodel=128,
                 layers=[5 * 128 * 256, 5 * 128 * 256 + 2 * 256 * 128],
                 device='cuda:0'):
        super(Model, self).__init__()
        self.device = device
        self.dmodel = dmodel
        self.len = seq_len
        print("len:", seq_len)
        print("this is GRU process!")
        self.rnn = RNNImputation()

        receptive_field_shape = seq_len//4
        layer_parameter_list = generate_layer_parameter_list(1, receptive_field_shape, layers, in_channel=in_features)
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)

        out_put_channel_number = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_number = out_put_channel_number + final_layer_parameters[1]

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(out_put_channel_number, out)

    def forward(self, x_input):
        (new_input, prediction) = self.rnn(x_input, self.device)
        x = new_input.permute(1, 2, 0)  # [seq_len,bs,in] -> [bs,in,seq_len]

        x = self.net(x)
        
        x = self.gap(x).squeeze()

        x = self.fc(x)

        return x, prediction











