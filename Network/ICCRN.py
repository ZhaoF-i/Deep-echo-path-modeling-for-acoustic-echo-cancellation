import torch.nn as nn
import torch.fft
import torch
from einops import rearrange
from multiply_orders import *

class CFB(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(CFB,self).__init__()
        self.conv_gate      = nn.Conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True)
        self.conv_input     = nn.Conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True)
        self.conv           = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,1), stride=1, padding=(1,0), dilation=1, groups=1, bias=True)
        self.ceps_unit  = CepsUnit(ch=out_channels)
        self.LN0     = LayerNorm( in_channels,f=160) 
        self.LN1     = LayerNorm(out_channels,f=160) 
        self.LN2     = LayerNorm(out_channels,f=160) 
    def forward(self, x):
        g = torch.sigmoid(self.conv_gate(self.LN0(x)))
        x = self.conv_input(x)
        y = self.conv(self.LN1(g*x))
        y = y + self.ceps_unit(self.LN2((1-g)*x))
        return y 


class CepsUnit(nn.Module):
    def __init__(self, ch):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f  = CH_LSTM_F(ch*2, ch,  ch*2)
        self.LN  = LayerNorm(ch*2,f=81)

    def forward(self, x0):
        x0 = torch.fft.rfft(x0, 160, 2)
        x = torch.cat([x0.real,x0.imag], 1)
        x = self.ch_lstm_f(self.LN(x))
        x = x[:,:self.ch] +1j*x[:,self.ch:]
        x = x*x0
        x = torch.fft.irfft(x, 160, 2)
        return x 


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm,self).__init__()
        self.w=nn.Parameter(torch.ones(1,c,f,1))
        self.b=nn.Parameter(torch.rand(1,c,f,1)*1e-4)
    def forward(self, x):
        mean = x.mean([1,2],keepdim=True)
        std  = x.std([1,2],keepdim=True)
        x = (x-mean)/(std+1e-8) *self.w +self.b
        return x


class NET(nn.Module):
    def __init__(self, order=10 ,channels=20):
        super().__init__()
        self.act = nn.ELU()
        self.n_fft = 319 
        self.hop_length = 160 
        self.window = torch.hamming_window(self.n_fft)
        self.order = order

        self.in_ch_lstm  = CH_LSTM_F(4, channels,  channels)
        self.in_conv     = nn.Conv2d(in_channels=4+channels, out_channels=channels, kernel_size=(1,1))
        self.cfb_e1 = CFB(channels, channels)
        # self.cfb_e2 = CFB(channels, channels)
        # self.cfb_e3 = CFB(channels, channels)
        # self.cfb_e4 = CFB(channels, channels)
        # self.cfb_e5 = CFB(channels, channels)
               
        self.ln      = LayerNorm(channels,160)
        self.ch_lstm = CH_LSTM_T(in_ch=channels, feat_ch=channels*2, out_ch=channels, num_layers=2)

        self.cfb_d1 = CFB(1*channels, channels)
        # self.cfb_d4 = CFB(2*channels, channels)
        # self.cfb_d3 = CFB(2*channels, channels)
        # self.cfb_d2 = CFB(2*channels, channels)
        # self.cfb_d1 = CFB(2*channels, channels)

        self.out_ch_lstm = CH_LSTM_T(2*channels, channels, channels*2)
        self.out_conv    = nn.Conv2d(in_channels=channels*3, out_channels=self.order*2, kernel_size=(1,1), padding=(0,0), bias=True)
        # self.sigmoid = nn.Sigmoid()

    def stft(self, x):
        b, m, t = x.shape[0], x.shape[1], x.shape[2],
        x = x.reshape(-1, t)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device))
        F, T = X.shape[1], X.shape[2]
        X = X.reshape(b, m, F, T, 2)
        X = torch.cat([X[..., 0], X[..., 1]], dim=1)
        return X

    def istft(self, Y, t):
        b,c,F,T=Y.shape
        m_out = int(c//2)
        Y_r = Y[:,:m_out]
        Y_i = Y[:,m_out:]
        Y = torch.stack([Y_r, Y_i], dim=-1)
        Y = Y.reshape(-1, F, T, 2)
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, length=t, window=self.window.to(Y.device))
        y = y.reshape(b, m_out, y.shape[-1])
        return y

    def forward(self, x):
        # x:[batch, channel, frequency, time]
        X0 = self.stft(x)
        mix_comp = torch.stack([X0[:, 0], X0[:, 2]], dim=1)
        far_comp = torch.stack([X0[:, 1], X0[:, 3]], dim=1)
        e0 = self.in_ch_lstm(X0)
        e0 = self.in_conv(torch.cat([e0,X0], 1))
        e1 = self.cfb_e1(e0)
        # e2 = self.cfb_e2(e1)
        # e3 = self.cfb_e3(e2)
        # e4 = self.cfb_e4(e3)
        # e5 = self.cfb_e5(e4)
                          
        lstm_out = self.ch_lstm(self.ln(e1))

        d1 = self.cfb_d1(torch.cat([e1 * lstm_out],dim=1))
        # d4 = self.cfb_d4(torch.cat([e4, d5],dim=1))
        # d3 = self.cfb_d3(torch.cat([e3, d4],dim=1))
        # d2 = self.cfb_d2(torch.cat([e2, d3],dim=1))
        # d1 = self.cfb_d1(torch.cat([e1, d2],dim=1))

        d0 = self.out_ch_lstm(torch.cat([e0, d1],dim=1))
        Y  = self.out_conv(torch.cat([d0, d1],dim=1))
        # b, c, f, t = Y.shape
        Y = Y.reshape(Y.shape[0], 2, self.order, Y.shape[2], Y.shape[3])
        estEchoPath = Y[:, :, :self.order]
        # mask = self.sigmoid(Y[:, :, -1])
        out = mix_comp - multiply_orders_(far_comp, estEchoPath, self.order)
        # out = out * mask

        y = self.istft(out, t=x.shape[-1])[:, 0]
        # far = self.istft(far_comp, t=x.shape[-1])[:, 0]

        return y, estEchoPath


class CH_LSTM_T(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if bi==False else 2
        self.linear = nn.Linear(self.bi*feat_ch,out_ch)
        self.out_ch = out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b,c,f,t = x.shape
        x  = rearrange(x, 'b c f t -> (b f) t c')
        x,_ = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b f) t c -> b c f t', b=b, f=f, t=t)
        return x

class CH_LSTM_F(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.linear= nn.Linear(2*feat_ch,out_ch)
        self.out_ch=out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b,c,f,t = x.shape
        x = rearrange(x, 'b c f t -> (b t) f c')   
        x,_  = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b t) f c -> b c f t', b=b, f=f, t=t)
        return x


def complexity():
    # inputs = torch.randn(1,1,16000)
    model = NET()

    check_causality_time_input(model, algo_lat=0)
    # output = model(inputs)
    # print(output.shape)

    from ptflops import get_model_complexity_info
    mac, param =  get_model_complexity_info(model, (2,16000), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(mac, param)
    '''
    1.93 GMac 463.1 k
    1.95 GMac 464.56 k
    '''

import numpy as np

def check_causality_time_input(model, sr=16000, algo_lat=0.005):
    """
    :param model: your DNN model in Torch, Tensorflow, etc.
    :param sr: sampling rate in Hz
    :param algo_lat: allowed algorithmic latency in seconds

    The idea is that we set the samples starting from a random position to NaN,
        and the DNN model that peeks the NaNs would output NaNs.
    """

    algo_lat = int(algo_lat*sr)
    sig_len_range = [2., 8.] # range of signal length in seconds

    R = 100
    for r in range(R):
        l = np.random.uniform(low=sig_len_range[0], high=sig_len_range[1])
        l = int(l*sr)
        sig = np.float32(np.random.randn(l))
        sig = sig / np.max(np.abs(sig)) * 0.9
        p = np.random.randint(len(sig))
        check_p = (p // 160 - 1) * 160 # 160为窗移
        sig[p:] = np.nan

        est_sig, _ = model(torch.stack([torch.tensor(sig), torch.tensor(sig)], dim=0).unsqueeze(0)) # obtain separation results using your model
        est_sig = est_sig[0].detach().numpy()
        assert est_sig.shape == sig.shape # they should have same length

        if p-algo_lat+1 >= 1 and np.sum(np.isnan(est_sig[:check_p-algo_lat+1])) > 0:
            print('For example %d, your model does NOT satisfy the algorithmic latency requirement!'%r)
            return

    print('Your model satisfies the algorithmic latency requirement!')

if __name__ == '__main__':
    complexity()


