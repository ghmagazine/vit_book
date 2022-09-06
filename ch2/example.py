# -*- coding: utf-8 -*-

# ----------------------------
# 必要なライブラリをインポート
# ----------------------------
import torch
import torch.nn as nn


# ----------------------------
# 2-1 準備
# ----------------------------
print("=======2-1 準備=======")

class SimpleMlp(nn.Module):
    def __init__(self, vec_length:int=16, hidden_unit_1:int=8, hidden_unit_2:int=2): 
        """
        引数:
            vec_length: 入力ベクトルの長さ 
            hidden_unit_1: 1つ目の線形層のニューロン数 
            hidden_unit_2: 2つ目の線形層のニューロン数
        """
        # 継承しているnn.Moduleの__init__()メソッドの呼び出し 
        super(SimpleMlp, self).__init__()
        # 1つ目の線形層
        self.layer1 = nn.Linear(vec_length, hidden_unit_1)
        # 活性化関数のReLU
        self.relu = nn.ReLU()
        # 2つ目の線形層
        self.layer2 = nn.Linear(hidden_unit_1, hidden_unit_2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝搬は、線形層→ReLU→線形層の順番 
        引数:
            x: 入力。(B, D_in)
                B: バッチサイズ、 D_in: ベクトルの長さ
        返り値:
            out: 出力。(B, D_out)
                B: バッチサイズ、 D_out: ベクトルの長さ 
        """
        # 1つ目の線形層
        out = self.layer1(x)
        # ReLU
        out = self.relu(out)
        # 2つ目の線形層
        out = self.layer2(out) 
        return out

vec_length = 16 # 入力ベクトルの長さ 
hidden_unit_1 = 8 # 1つ目の線形層のニューロン数 
hidden_unit_2 = 2 # 2つ目の線形層のニューロン数

batch_size = 4 # バッチサイズ。入力ベクトルの数 

# 入力ベクトル。xの形状: (4, 16)
x = torch.randn(batch_size, vec_length)
# MLPを定義
net = SimpleMlp(vec_length, hidden_unit_1, hidden_unit_2) 
# MLPで順伝搬
out = net(x)
# MLPの出力outの形状が(4, 2)であることを確認 
print(out.shape)