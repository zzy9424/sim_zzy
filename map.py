import numpy as np
from PIL import Image
# 读取图像
image_path = 'officemap.pgm'
image = Image.open(image_path)
# 将图像转换为numpy数组
image_array = np.array(image)
# 设置numpy打印选项，显示整个数组
with np.printoptions(threshold=np.inf):
    print(image_array)