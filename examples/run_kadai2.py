import sys
import os

# hopfield ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hopfield.network import HopfieldNetwork
from hopfield.utils import add_noise, judge_accuracy, calculate_similarity
from data.data import A, B, C, D, E, F, H, Z

import numpy as np
import matplotlib.pyplot as plt
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
        
        # ホップフィールドネットワークで想起
        output = model.recall_asyn(noisy_input)

        # 正解判定・類似度評価
        accurate_num += judge_accuracy(output, true_pattern)
        similarity_sum += calculate_similarity(output, true_pattern)

    accurate_rate = accurate_num / trials
    similarity_degree = similarity_sum / trials

    print(f"--- 画像の種類:{n_patterns}, ノイズ:{noise_rate} ---")
    print(f"正確率: {accurate_rate:.2f}")
    print(f"平均類似度: {similarity_degree:.2f}")
    print()

    return accurate_rate, similarity_degree

def plot_performance(n_patterns, step=0.05):
    noise_rates = np.arange(0.0, 1.01, step)
    accuracies = []
    similarities = []

    for rate in noise_rates:
        acc, sim = evaluate_model(n_patterns=n_patterns, noise_rate=rate)
        accuracies.append(acc)
        similarities.append(sim)

    plt.plot(noise_rates, accuracies, label=f'Accuracy (n={n_patterns})', marker='o')
    plt.plot(noise_rates, similarities, label=f'Similarity (n={n_patterns})', marker='x')

# 描画
plt.figure(figsize=(10, 6))
plot_performance(n_patterns=2)
plot_performance(n_patterns=4)
plt.xlabel("Noise Rate")
plt.ylabel("Accuracy / Similarity")
plt.title("Hopfield Network Performance vs Noise Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('kadai2.png')
print("saved as kadai2.png")
plt.show()

