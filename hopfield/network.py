import numpy as np
from numpy.random import *

class HopfieldNetwork:
    def __init__(self):
        self.weights = None # 重み行列。
        self.theta = 0 # 閾値。全て0としている。

    # トレーニング時の関数。Autocorrelation。
    def train(self, patterns): # patterns: 入力画像をまとめたもの。
        n_units = patterns.shape[1] # n_units: 画素の数。今回の場合は5×5の画像なので、25。25個のノードを作る。
        self.weights = np.zeros((n_units, n_units)) # 重み行列は25×25となる。
        for p in patterns: # patterns内の各々の入力画像について行う。
            self.weights += np.outer(p, p) # それぞれの入力画像について画素同士の相関を計算し、足していく。
        np.fill_diagonal(self.weights, 0) # 行列の対角成分を強制的に0にする。(重みwii = 0)
        self.weights /= len(patterns) # 最後に、入力画像の数で割る。

    # 同期更新の関数。ステップ数を決めて回す。
    def recall_syn(self, pattern, steps=20):
        out = pattern.copy() # 入力画像のコピーを作る。
        for _ in range(steps): # stepsの数だけループを回す。
            out = np.sign(self.weights @ out) # 同期更新。全体を一気に更新。
        return out
    
    # 同期更新の関数。ポテンシャルエネルギーが変化しなくなるまで回す。
    def recall_syn_usePotentialEnergy(self, pattern):
        out = pattern.copy() # 入力画像のコピーを作る。
        e_old = float("inf")
        for _ in range(10000): # ポテンシャルエネルギーの更新がなくなるまで更新。ここでは、10000ステップを上限に設定。
            out = np.sign(self.weights @ out) # 同期更新。全体を一気に更新。
            e_new = self.energy(pattern)
            # エネルギーが変化しなくなったら打ち切り
            if e_new == e_old:
                break
            e_old = e_new
        return out
    
    # 非同期更新
    def recall_asyn(self, pattern, steps=300):
        out = pattern.copy() # 入力画像のコピーを作る。
        for _ in range(steps): # stepsの数だけループを回す。
            num = randint(len(out)) # ランダムに一つのニューロンに注目。
            out[num] = np.sign(np.dot(self.weights[num], out)) # 注目したニューロンを更新。
        return out

    # ポテンシャルエネルギー計算の関数
    def energy(self, x):
        # V = -1/2 * ΣΣ w*x*x + Σ θ*x 
        return -0.5*np.dot(x.T, np.dot(self.weights, x)) + self.theta * np.sum(x)
