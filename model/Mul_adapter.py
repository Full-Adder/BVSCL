import torch
import torch.nn as nn
import torch.nn.functional as F

class MulAdapter(nn.Module):
    def __init__(self, input_size, learning_rate_task):
        super(MulAdapter, self).__init__()
        self.cov1 = nn.Conv3d(input_size, input_size, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.cov2 = nn.Conv3d(input_size, input_size, kernel_size=1, stride=1, padding=0)

        self.q = nn.Linear(input_size, input_size)
        self.k = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)
        self.d_dim = 1/torch.sqrt(torch.tensor(input_size))

        self.learning_rate = learning_rate_task

    def forward(self, x, task=-1):
        x = self.cov1(x)
        x = self.relu(x)
        x = self.cov2(x)

        s = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        q = self.q(s)
        k = self.k(s)
        v = self.v(s)
        if task == -1:
            att = F.softmax(q@k.transpose(1,2)/self.d_dim, dim=-1)
        else:
            att = F.softmax(q@k.transpose(1,2)/self.d_dim, dim=-1)*self.learning_rate[task]
        att = v.transpose(1,2)@att
        att = att.view(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4))

        return x + att


if __name__ == '__main__':
    x = torch.randn(1,768,16,12,7)
    mm = MulAdapter(768,768)
    re = mm(x)
    print(re.shape)