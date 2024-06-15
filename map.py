import cv2
import numpy as np
from PIL import Image


def find_black_rectangles(image_array, threshold=85):
    # 将灰度图像二值化，黑色为0，其它为255
    _, binary_image = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        rectangles.append((top_left, bottom_right))

    return rectangles


def draw_rectangles(image_array, rectangles):
    # 将灰度图像转换为RGB图像
    image_color = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    # 画出每个矩形
    for rect in rectangles:
        top_left, bottom_right = rect
        # 生成随机颜色
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.rectangle(image_color, top_left, bottom_right, color, 2)  # 随机颜色的矩形，线宽为2

    return image_color

def crop_gray_border(image_array, gray_threshold=200,margin=10):
    # 找到上边界
    for top in range(image_array.shape[0]):
        if np.mean(image_array[top, :]) < gray_threshold:
            break

    # 找到下边界
    for bottom in range(image_array.shape[0] - 1, -1, -1):
        if np.mean(image_array[bottom, :]) < gray_threshold:
            break

    # 找到左边界
    for left in range(image_array.shape[1]):
        if np.mean(image_array[:, left]) < gray_threshold:
            break

    # 找到右边界
    for right in range(image_array.shape[1] - 1, -1, -1):
        if np.mean(image_array[:, right]) < gray_threshold:
            break

    # 裁剪图像
    cropped_image_array = image_array[top-margin:bottom+margin+1, left-margin:right+margin+1]
    return cropped_image_array
# 读取图像
image_path = 'pngmap.png'
if image_path.endswith('pgm'):
    image = Image.open(image_path)
elif image_path.endswith('png'):
    image = Image.open(image_path).convert('L')  # 将图像转换为灰度图像

# 将图像转换为numpy数组
image_array = np.array(image)
# 设置numpy打印选项，显示整个数组
# with np.printoptions(threshold=np.inf):
#     print(image_array)
print(len(image_array),len(image_array[0]))

# 裁剪掉周围一圈灰色的边
image_array = crop_gray_border(image_array)

# 查找黑色区域的矩形
rectangles = find_black_rectangles(image_array)

# 打印结果
for rect in rectangles:
    print(f"Top-left: {rect[0]}, Bottom-right: {rect[1]}")

# 绘制矩形到图像上
image_with_rectangles = draw_rectangles(image_array, rectangles)

# 保存结果图像
output_path = 'output_image_with_rectangles.jpg'
cv2.imwrite(output_path, image_with_rectangles)
print(f"结果图像已保存到 {output_path}")