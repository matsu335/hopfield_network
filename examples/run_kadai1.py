import sys
import os

# hopfield ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hopfield.network import HopfieldNetwork
from hopfield.utils import add_noise, judge_accuracy, calculate_similarity
from data.data import A, B, C, D, E, F, H, Z

import numpy as np
import random

patterns = [A, B, C, F, H, Z]

# 性能を評価する関数。n_patterns: 画像が何種類あるか。noise_rate: ノイズの強さ。trials: 試行回数。
def evaluate_model(n_patterns, noise_rate=0, trials=100):
    selected_patterns = patterns[:n_patterns]

    model = HopfieldNetwork()
    model.train(np.array(selected_patterns))

    accurate_num = 0
    similarity_sum = 0

    for _ in range(trials):
        # ランダムに 1 つのパターンを選択
        idx = random.randint(0, n_patterns - 1)
        true_pattern = selected_patterns[idx]

        # ノイズを加える
        noisy_input = add_noise(true_pattern, noise_rate)
        
        # ホップフィールドネットワークで想起。実験3ではこの想起方法を変える。
        output = model.recall_asyn(noisy_input)
        # output = model.recall_syn(noisy_input)
        # output = model.recall_syn_usePotentialEnergy(noisy_input)

        # 正解判定・類似度評価
        accurate_num += judge_accuracy(output, true_pattern)
        similarity_sum += calculate_similarity(output, true_pattern)

    accurate_rate = accurate_num / trials
    similarity_degree = similarity_sum / trials

    print(f"--- 画像の種類:{n_patterns}, ノイズ:{noise_rate} ---")
    print(f"正確率: {accurate_rate:.2f}")
    print(f"平均類似度: {similarity_degree:.2f}")
    print()

evaluate_model(1, 0.05)
evaluate_model(1, 0.1)
evaluate_model(1, 0.15)
evaluate_model(1, 0.2)
evaluate_model(2, 0.05)
evaluate_model(2, 0.1)
evaluate_model(2, 0.15)
evaluate_model(2, 0.2)
evaluate_model(3, 0.05)
evaluate_model(3, 0.1)
evaluate_model(3, 0.15)
evaluate_model(3, 0.2)
evaluate_model(4, 0.05)
evaluate_model(4, 0.1)
evaluate_model(4, 0.15)
evaluate_model(4, 0.2)
evaluate_model(5, 0.05)
evaluate_model(5, 0.1)
evaluate_model(5, 0.15)
evaluate_model(5, 0.2)
evaluate_model(6, 0.05)
evaluate_model(6, 0.1)
evaluate_model(6, 0.15)
evaluate_model(6, 0.2)
