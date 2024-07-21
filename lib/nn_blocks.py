from copy import deepcopy
from collections import OrderedDict

from torch import (
    arange as torch_arange,
    cat as torch_cat,
    einsum as torch_einsum,
    finfo as torch_finfo,
    nan_to_num as torch_nan_to_num,
)
from torch.nn import (
    Dropout as torch_drop_out,
    LayerNorm as torch_layer_norm,
    Linear as torch_linear,
    MultiheadAttention as torch_multi_head_attention,
    Module as torch_module,
    ModuleList as torch_module_list,
)
from torch.nn.functional import (
    relu as torch_relu,
    pad as torch_pad,
    )
import torch
from torch import nn
import torchvision.models

def linear_block(in_features, out_features, activation='relu', drop_p = 0):
    if activation != 'None':
        activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()], 
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()]
        ])
        return nn.Sequential (OrderedDict([
        ('drop_out', nn.Dropout(p=drop_p)),
        ('fc', nn.Linear(in_features, out_features)),
        ('activation' ,activations[activation]),
        ]))
    else:
        return nn.Sequential(OrderedDict([('fc',nn.Linear(in_features, out_features))]))

class recurrent_block(nn.Module):
    def __init__(self, in_features, hidden_size, device, return_final_hidden = False, recurrence_cell = 'rnn', num_recurrent_layers=1, rnn_non_linearity='tanh', bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.device = device
        self.hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers
        self.return_final_hidden = return_final_hidden
        self.recurrence_cell = recurrence_cell
        if recurrence_cell == 'rnn':
            self.rnn = nn.RNN(input_size = in_features, 
                     hidden_size = hidden_size, 
                     batch_first = True,
                     num_layers = num_recurrent_layers,
                     nonlinearity = rnn_non_linearity,
                     bidirectional=self.bidirectional)
        elif recurrence_cell == 'gru':
            self.rnn = nn.GRU(input_size = in_features, 
                        hidden_size = hidden_size, 
                        batch_first = True,
                        num_layers = num_recurrent_layers,
                        bidirectional=self.bidirectional)
        elif recurrence_cell == 'lstm':
            self.rnn = nn.LSTM(input_size = in_features, 
                        hidden_size = hidden_size, 
                        batch_first = True,
                        num_layers = num_recurrent_layers,
                        bidirectional=self.bidirectional)
        else:
            raise NotImplementedError
        self.to(device)
        
    def init_hidden(self,batch_size):
        if self.recurrence_cell == 'lstm':
            if self.bidirectional:
                rnn_hidden =torch.randn(self.num_recurrent_layers*2, batch_size, self.hidden_size)
                c_hidden = torch.randn(self.num_recurrent_layers*2, batch_size, self.hidden_size)
            else:
                rnn_hidden = torch.randn(self.num_recurrent_layers, batch_size, self.hidden_size)
                c_hidden = torch.randn(self.num_recurrent_layers, batch_size, self.hidden_size)
            rnn_hidden = rnn_hidden.to(self.device)
            c_hidden = c_hidden.to(self.device)
            return (rnn_hidden, c_hidden)
        else:
            if self.bidirectional:
                rnn_hidden =torch.randn(self.num_recurrent_layers*2, batch_size, self.hidden_size)
            else:
                rnn_hidden = torch.randn(self.num_recurrent_layers, batch_size, self.hidden_size)
            # TODO: Need to be moved to wherever model coef are
            rnn_hidden = rnn_hidden.to(self.device)
        return rnn_hidden   
    def forward(self, x, seq_len):
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True)
        x, last_hidden = self.rnn(x, hidden)
        if not self.return_final_hidden:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = x.contiguous()
            return x
        else:
            return last_hidden.squeeze(0)

def pre_trained_resnet_18_backend_block(num_features, no_head=False):
    model_ft = torchvision.models.resnet18(pretrained=True)
    if no_head:
        model_ft = torch.nn.Sequential(*(list(model_ft.children())[:-1]))
    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_features)
    return model_ft

class pre_trained_Transformer_subword_embedder_block(nn.Module):
    def __init__(self, model, filter_special_tokens_fn=None):
        super().__init__()
        self.model = model.eval()
        self.filter_special_tokens_fn = filter_special_tokens_fn
    def forward(self, x, sub_word_idx):
        with torch.no_grad():
            last_hidden_state, pooler_output = self.model(x)
            if self.filter_special_tokens_fn:
                last_hidden_state = self.filter_special_tokens_fn(last_hidden_state)
                ret_tensor_lst = []
                for i in range(x.shape[0]):
                    ret_tensor_lst.append(last_hidden_state[i,sub_word_idx[i],:])
                ret_state = torch.vstack(ret_tensor_lst)
                return  ret_state

class fc_classifier(nn.Module):
    """Classifier with configurable fully connected layers"""
    def __init__(self, input_dim, linear_block_sizes, output_dim, linear_block_activations, drop_ps):
        super().__init__()
        linear_block_sizes = [input_dim, *linear_block_sizes, output_dim]
        lst = [(f'linear_block{i}', linear_block(in_f, out_f, activation, drop_p)) for i, (in_f, out_f, activation, drop_p) in 
              enumerate(zip(linear_block_sizes,linear_block_sizes[1:], linear_block_activations, drop_ps))]
        self.linear_blocks = nn.Sequential(OrderedDict(lst))
    def forward(self, x):
        return self.linear_blocks(x)

class recurrence_classifier(nn.Module):
    seq_sample_label=True
    def __init__(self, pre_recurrence_block, recurrence_block, post_recurrence_block):
        super().__init__()
        self.pre_recurrence  = pre_recurrence_block
        self.recurrence =  recurrence_block 
        self.post_recurrence = post_recurrence_block
    @classmethod
    def batch_seqs_by_sort_and_pad (cls, batch, return_Y=True, seq_classification=False):
        # Get batch_size
        batch_size = len(batch)
        if return_Y:
            # Get the dimensionality of the data
            data_input_dim = batch[0][0].shape[1:]
            # Get the sequence lengthes
            seq_lens = [x[0].shape[0] for x in batch]
            device = batch[0][0].device
        else:
            data_input_dim = batch[0].shape[1:]
            seq_lens = [x.shape[0] for x in batch]
            device = batch[0].device
        # Get sorting indices
        sorted_indices = sorted(list(range(len(seq_lens))), key=lambda i: seq_lens[i], reverse=True)
        # sort the sequence lengthes based on the sorting indices
        seq_lens = [seq_lens[i] for i in sorted_indices]
        # sort the batches on the sorting indices
        batch = [batch[i] for i in sorted_indices]
        # Get the max length
        max_len = max(seq_lens)
        # Create place holders for the returned batch
        batch_x = torch.zeros((batch_size, max_len, *data_input_dim))
        if return_Y:
            if cls.seq_sample_label:
                batch_y = torch.ones((batch_size, max_len),dtype=torch.long)*-1
            else:
                batch_y = torch.ones((batch_size),dtype=torch.long)*-1
        # Copy the data
        for ind, curr_len in enumerate(seq_lens):
            if return_Y:
                if cls.seq_sample_label:
                    batch_y[ind,:curr_len] = batch[ind][1]
                else:
                    batch_y[ind] = batch[ind][1]
            batch_x[ind,:curr_len,:] = batch[ind][0]
        batch_x = batch_x.to(device)
        if return_Y:
            batch_y = batch_y.to(device)
            return ((batch_x,seq_lens,sorted_indices), batch_y, [curr_batch[2] for curr_batch in batch])
        else:
            return ((batch_x,seq_lens,sorted_indices))

class seq_samples_classifier(recurrence_classifier):
    def __init__(self, pre_recurrence_block, recurrence_block, post_recurrence_block):
        super().__init__(pre_recurrence_block, recurrence_block, post_recurrence_block)
    def forward(self,x):
        seq_len = x[1]
        x = x[0]
        x = self.pre_recurrence(x)
        x = self.recurrence(x,seq_len)
        x = self.post_recurrence(x)
        return torch.transpose(x,1,2)

class seq_samples_classifier_with_shortcut(recurrence_classifier):
    '''label samples in a sequence'''
    def __init__(self, pre_recurrence_block, recurrence_block, post_recurrence_block, shortcut_block):
      super().__init__(pre_recurrence_block, recurrence_block, post_recurrence_block)
      self.shortcut_block = shortcut_block
    def forward(self,x):
        seq_len = x[1]
        x = x[0]
        x_shortcut = self.shortcut_block(x)
        x = self.pre_recurrence(x)
        x = self.recurrence(x,seq_len)
        x = torch.cat([x, x_shortcut], dim=2)
        x = self.post_recurrence(x)
        return torch.transpose(x,1,2)

class seq_classifier(recurrence_classifier):
    seq_sample_label=False
    '''label samples in a sequence'''
    def __init__(self, pre_recurrence_block, recurrence_block, post_recurrence_block, pass_seq_len_to_pre_recurrence=False):
        super().__init__(pre_recurrence_block, recurrence_block, post_recurrence_block)
        recurrence_block.return_final_hidden = True
        self.pass_seq_len_to_pre_recurrence = pass_seq_len_to_pre_recurrence
    def forward(self,x):
        seq_len = x[1]
        x = x[0]
        if self.pass_seq_len_to_pre_recurrence:
            x = self.pre_recurrence(x,seq_len)
        else:
            x = self.pre_recurrence(x)
        x = self.recurrence(x,seq_len)
        x = self.post_recurrence(x)
        return x

class pre_recurrence_seq_conv(nn.Module):
    def __init__(self, CONV_BLOCK, CONV_DIM):
        super().__init__()
        self.CONV_BLOCK = CONV_BLOCK
        self.CONV_DIM = CONV_DIM
    def forward(self, x, seq_len):
        with torch.no_grad():
            [batch_size,seq_len, channels,h,w] = x.shape
        x = x.view(-1, channels, h, w)
        x = self.CONV_BLOCK(x)
        x = x.view(batch_size, seq_len, self.CONV_DIM)
        return (x)

class sub_word_classifier(nn.Module):
    def __init__(self, embedder, head_classifier):
        super().__init__()
        self.embedder = embedder
        self.head_classifier = head_classifier
    def forward(self, x):
        sub_word_idx = x[1]
        x = x[0]
        x = self.embedder(x, sub_word_idx)
        x = self.head_classifier(x)
        return x

def look_arnd(x, pad_value):
    backward, forward, dim = 1, 1, 2
    windows = x.shape[1]
    dims = (len(x.shape) - dim) *(0, 0)
    padded_x = torch_pad(x, (*dims, backward, forward),\
        value = pad_value)
    tensors = [padded_x[:, ind:(ind + windows), ... ]\
        for ind in range(forward + backward + 1)]
    return torch_cat(tensors, dim = dim)

class local_bidir_attn(torch_module):
    def __init__(self,
                window_size,
                dropout,):
        super().__init__()
        self.window_size = window_size
        self.drop_out = torch_drop_out(dropout)
    def forward(self, q, k, v, padding_mask = None):
        if padding_mask is not None:
            padding_mask = (~ padding_mask)
        # collection info regarding the batch
        batch_size, context_len, sample_dim, device, dtype =\
            q.shape[0], q.shape[1], q.shape[2], q.device,\
                q.dtype
        assert context_len % self.window_size == 0, f"Context\
len {context_len} must be divisible by window size\
window_size {self.window_size}"
        windows = context_len // self.window_size
        ticker = torch_arange(context_len, device = device,\
            dtype = dtype)
        ticker = ticker.reshape(1, windows, self.window_size)
        bucket_fn = lambda t: t.reshape(batch_size, windows,\
            self.window_size, -1)
        bucketed_q, bucketed_k, bucketed_v = map(bucket_fn, (q, k, v))
        bucketed_k = look_arnd(bucketed_k, pad_value = -1)
        dots = torch_einsum('bhie,bhje->bhij', bucketed_q,\
            bucketed_k) * (sample_dim ** -0.5)
        bucketed_ticker = look_arnd(ticker, pad_value = -1)
        mask = bucketed_ticker[:, :, None, :] == -1
        dots.masked_fill_(mask, - torch_finfo(dots.dtype).max)
        del mask
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, windows,\
                        self.window_size)
            mask_query = mask_key = padding_mask
            mask_key = look_arnd(x = mask_key, pad_value = False)
            mask = (mask_query[:, :, :, None] \
                * mask_key[:, :, None, :])
            dots.masked_fill_(~mask, -torch_finfo(dots.dtype).max)
        attn = dots.softmax(dim=-1)
        attn = self.drop_out(attn)
        bucketed_v = look_arnd(bucketed_v, pad_value = -1)
        out = torch_einsum("bhij,bhje->bhie", attn, bucketed_v)
        out = out.reshape(-1, context_len, sample_dim)
        return out

def _get_clones(module, n):
    return torch_module_list([deepcopy(module)\
        for i in range(n)])
class transformer_encoder_layer(torch_module):
    def __init__(self,
                d_model,
                nhead,
                dim_feedforward = 2048,
                dropout = 0.1,
                activation = torch_relu,
                layer_norm_eps = 1e-5,
                window_sz = None,
                device = None,):
        factory_kwargs = {'device': device}
        super().__init__()
        self.window_sz = window_sz
        if self.window_sz is None:
            self.self_attn = torch_multi_head_attention(d_model,
                            nhead,
                            dropout = dropout,
                            batch_first = True,
                            **factory_kwargs)
        else:
            self.self_attn = local_bidir_attn(self.window_sz,\
                                            dropout = dropout)
        self.linear1 = torch_linear(d_model, dim_feedforward,
                            **factory_kwargs)
        self.dropout = torch_drop_out(dropout)
        self.linear2 = torch_linear(dim_feedforward, d_model,
                                    **factory_kwargs)
        self.norm1 = torch_layer_norm(d_model,
                        eps = layer_norm_eps, **factory_kwargs)
        self.norm2 = torch_layer_norm(d_model,
                        eps = layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch_drop_out(dropout)
        self.dropout2 = torch_drop_out(dropout)
        self.activation = activation
    def _sa_block(self, x, padding_mask):
        if self.window_sz is None:
            x = self.self_attn(x, x, x,\
                            attn_mask = None,\
                            key_padding_mask = padding_mask,\
                            need_weights = False)[0]
        else:
            x = self.self_attn(x, x, x,\
                padding_mask = padding_mask)
        return self.dropout1(x)
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(\
            self.linear1(x))))
        return self.dropout2(x)
    def forward(self, src, padding_mask = None):
        x = src
        x = self.norm1(x + self._sa_block(x, padding_mask))
        x = self.norm2(x + self._ff_block(x))
        x = torch_nan_to_num(x)
        return x
class transformer_encoder(torch_module):
    def __init__(self, encoder_layer,
                num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
    def forward(self, src, padding_mask = None):
        output = src
        for mod in self.layers:
            output = mod(src, padding_mask = padding_mask)
        return output