import torch
from torch.nn.init import kaiming_normal_
import pandas as pd

y = torch.tensor([[[1,2,3],
                   [4,5,6],
                   [7,8,9]],

                  [[2,4,6],
                   [8,10,2],
                   [4,6,8]]])  # [2,3,3]

i = torch.tensor([[[0,2,1]],[[2,1,1]]])  # [2,1,3]


print(torch.gather(y, dim=1, index=i))