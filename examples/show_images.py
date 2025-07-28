
import sys
import os

# hopfield ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hopfield.utils import show_images
from data.data import A, B, C, D, E, F, H, Z

images = [A, B, C, D, E, F, H, Z]
titles = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'Z']
show_images(images, titles, scale=5)