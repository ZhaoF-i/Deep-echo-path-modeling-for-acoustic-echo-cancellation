import torch
import torch.nn.functional as F

def multiply_orders(far, estEchoPath, order):
    # input shape: B,C,T,F
    padding_far = F.pad(far, (0, 0, order-1, 0))
    t = far.shape[2]
    estEcho = []
    for i in range(order):
        estEcho.append(padding_far[:, :, i:i+t, :] * estEchoPath[:, :, i, :, :])
    return torch.stack(estEcho, dim=2).sum(dim=2)
def multiply_orders_(far, estEchoPath, order):
    # input shape: B,C,F,T
    padding_far = F.pad(far, (order-1, 0, 0, 0))
    t = far.shape[3]
    estEcho = []
    for i in range(order):
        rr = padding_far[:, 0, :, i:i+t] * estEchoPath[:, 0, i, :, :]
        ri = padding_far[:, 0, :, i:i+t] * estEchoPath[:, 1, i, :, :]
        ir = padding_far[:, 1, :, i:i+t] * estEchoPath[:, 0, i, :, :]
        ii = padding_far[:, 1, :, i:i+t] * estEchoPath[:, 1, i, :, :]
        r = rr - ii
        i = ri + ir
        estEcho.append(torch.stack([r,i], dim=1))
    return torch.stack(estEcho, dim=2).sum(dim=2)