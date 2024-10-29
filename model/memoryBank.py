import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F

class MemoryBank(nn.Module):

    def cosine_similarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = F.normalize(tensor_1, p=2, dim=0)
        normalized_tensor_2 = F.normalize(tensor_2, p=2, dim=0)
        cosine_sim = torch.sum(normalized_tensor_1 * normalized_tensor_2)
        return cosine_sim

    def __init__(self, name='0', pth_dir=None):
        super(MemoryBank, self).__init__()
        self.name = name
        if pth_dir is None:
            self.memory_bank = dict()
        else:
            self.memory_bank = pickle.load(open(pth_dir+f'/{name}.pkl', 'rb'))

    def add(self, x, target=None):
        if target is not None:
            target = target.view(1,1,1,-1)
            x = x.view(1,1,1,-1)
            target = F.interpolate(target, x.size()[-2:], mode='bilinear', align_corners=True)
            target = target.view(1, -1)
            x = x.view(1, -1)
            flg = False
            if len(self.memory_bank) == 0:
                self.memory_bank[x] = target
                return x
            for i, v in self.memory_bank.items():
                cos = self.cosine_similarity(i, x)
                if cos < 0.3:
                    flg = True
                    del self.memory_bank[i]
                    key = (0.5+cos/2)*x + cos/2*i
                    val = (0.5+cos/2)*v + cos/2*target
                    self.memory_bank[key] = val
                    return 0.3*val+0.7*x
            if flg == False or len(self.memory_bank) == 0:
                self.memory_bank[x] = target
                return x
        else:
            if len(self.memory_bank) == 0:
                return x
            for i, v in self.memory_bank.items():
                cos = self.cosine_similarity(i, x)
                if cos < 0.3:
                    return 0.3 * self.memory_bank[i] + 0.7 * x

    def forward(self, x, target=None):
        re = []
        re_batch = []
        xx = x.view(x.size(0), -1)
        if target is not None:
            tt = target.view(target.size(0), -1)
            for batch_id in range(x.size(0)):
                re_batch.append(self.add(xx[batch_id], tt[batch_id]))
            re_batch = torch.stack(re_batch, dim=0)
            # re.append(re_batch)
            return re_batch.view(x.size())
        else:
            for batch_id in range(x.size(0)):
                re_batch.append(self.add(xx[batch_id]))
            re_batch = torch.stack(re_batch, dim=0)
            # re.append(re_batch)
            return re_batch.view(x.size())

    def save(self):
        pickle.dump(self.memory_bank, open(f'./MB/{self.name}.pkl', 'wb'))


if __name__ == '__main__':
    x = torch.randn((1,768,16,12,7))
    mm = MemoryBank()
    for i in range(20):
        mm.update(x, x)
    print(mm.memory_bank)

