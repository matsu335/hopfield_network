import sys
import os
import itertools

# hopfield ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hopfield.network import HopfieldNetwork
from hopfield.utils import add_noise, judge_accuracy, calculate_similarity
from data.data import A, B, C, D, E, F, H, Z

import numpy as np
import random

# 性能を評価する関数。patterns: 選ばれた画像のリスト。noise_rate: ノイズの強さ。trials: 試行回数。
def evaluate_model(patterns, patterns_character, noise_rate=0.1, trials=100):
    selected_patterns = patterns

    model = HopfieldNetwork()
    model.train(np.array(selected_patterns))

    accurate_num = 0
    similarity_sum = 0

    for _ in range(trials):
        # ランダムに 1 つのパターンを選択
        idx = random.randint(0, 2)
        true_pattern = selected_patterns[idx]

        # ノイズを加える
        noisy_input = add_noise(true_pattern, noise_rate)
        
        # ホップフィールドネットワークで想起。
        output = model.recall_asyn(noisy_input)

        # 正解判定・類似度評価
        accurate_num += judge_accuracy(output, true_pattern)
        similarity_sum += calculate_similarity(output, true_pattern)

    accurate_rate = accurate_num / trials
    similarity_degree = similarity_sum / trials

    return {
        "patterns": patterns_character,
        "accurate_rate": accurate_rate,
        "similarity_degree": similarity_degree,
    }

# 使用する文字パターンとラベルの辞書
pattern_dict = {
    'A': A, 'B': B, 'C': C, 'D': D,
    'E': E, 'F': F, 'H': H, 'Z': Z
}

# 全ての3つの組み合わせを作成
combinations = list(itertools.combinations(pattern_dict.keys(), 3))

# 全ての組み合わせでモデルを評価
results = []
for combo in combinations:
    patterns = [pattern_dict[ch] for ch in combo]
    label = "[" + ", ".join(combo) + "]"
    result = evaluate_model(patterns, label)
    results.append(result)

# 正解率の高い順にソート
results.sort(key=lambda x: x["accurate_rate"], reverse=True)

# 表示
print("パターン,正確率,平均類似度")
for r in results:
    pattern_str = r['patterns'].replace('[', '').replace(']', '').replace(' ', '').replace(',', '-')
    print(f"{pattern_str},{r['accurate_rate']:.2f},{r['similarity_degree']:.2f}")