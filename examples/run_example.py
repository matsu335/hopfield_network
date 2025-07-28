import sys
import os

# hopfield ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hopfield.network import HopfieldNetwork
from hopfield.utils import show_images, add_noise
from data.data import A, C

import numpy as np

# パターンの定義（例: 3x3）
# pattern1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])
# pattern2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1])
pattern1 = A
pattern2 = C

model = HopfieldNetwork()
model.train(np.array([pattern1, pattern2]))

# テストパターンのノイズ付き入力
# noisy_input = np.array([1, -1, 1, -1, -1, -1, 1, -1, 1])
noisy_input = add_noise(A, 1)
output = model.recall_asyn(noisy_input)

print("Recalled pattern:", output)

images = [pattern1, pattern2, noisy_input, output]
titles = ['training1', 'training2', 'input', 'output']
show_images(images, titles, scale=5)