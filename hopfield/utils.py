import numpy as np
import matplotlib.pyplot as plt

# def array_to_image(pattern, scale=5):
#     image = pattern.reshape((scale,scale))

#     # 画像として表示
#     plt.imshow(image, cmap='gray', interpolation='nearest')
#     plt.axis('off')  # 軸を非表示にする
#     plt.show()

def show_images(image_arrays, titles, scale=5):
    images = [img.reshape((scale, scale)) for img in image_arrays]
    # 画像の枚数
    num_images = len(images)

    # サブプロットを作成（横に並べる）
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.5, 3))

    # 画像とタイトルをそれぞれのサブプロットに描画
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray', interpolation='nearest')
        ax.set_title(titles[i])
        ax.axis('off')  # 軸を非表示

    plt.tight_layout()
    plt.show()

def add_noise(image_arrays, rate):
    image_arrays = np.array(image_arrays)  # 確実にndarrayに変換
    # 乱数マスクの作成: Trueの位置を反転させる
    mask = np.random.rand(*image_arrays.shape) < rate
    noisy_images = np.where(mask, -image_arrays, image_arrays)
    return noisy_images

# 完全再現したかどうかを判定。
def judge_accuracy(test_pattern, ans_pattern):
    if np.array_equal(test_pattern, ans_pattern):
        return 1
    else:
        return 0

# 正解との類似度を計算。
def calculate_similarity(test_pattern, ans_pattern):
    accurate_num = np.sum(test_pattern == ans_pattern)
    accurate_rate = accurate_num / len(test_pattern)
    return accurate_rate