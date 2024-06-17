import math

from PIL import Image
import yaml

def crop_gray_border(image,margin=10):
    # Convert image to grayscale
    gray_image = image.convert('L')

    # Get width and height of the image
    width, height = gray_image.size

    # Define threshold for what is considered as gray (adjustable)
    gray_threshold = 254  # Adjust this threshold based on your image characteristics

    # Analyze top border
    top_border = 0
    for y in range(height):
        if all(gray_image.getpixel((x, y)) < gray_threshold for x in range(width)):
            top_border = y + 1
        else:
            break

    # Analyze bottom border
    bottom_border = height
    for y in range(height - 1, -1, -1):
        if all(gray_image.getpixel((x, y)) < gray_threshold for x in range(width)):
            bottom_border = y
        else:
            break

    # Analyze left border
    left_border = 0
    for x in range(width):
        if all(gray_image.getpixel((x, y)) < gray_threshold for y in range(height)):
            left_border = x + 1
        else:
            break

    # Analyze right border
    right_border = width
    for x in range(width - 1, -1, -1):
        if all(gray_image.getpixel((x, y)) < gray_threshold for y in range(height)):
            right_border = x
        else:
            break

    # Crop the image using the identified borders
    cropped_image = image.crop((left_border-margin, top_border-margin, right_border+margin, bottom_border+margin))

    return cropped_image

def max_pixel_value(image, x):
    width, height = image.size
    num_x_regions = width // x
    num_y_regions = height // x

    new_image = Image.new(image.mode, (width, height),(255,))  # Create new image with the same mode

    for i in range(num_x_regions):
        for j in range(num_y_regions):
            # Calculate the bounds of the current x by x region
            left = i * x
            upper = j * x
            right = left + x
            lower = upper + x

            # Find maximum pixel value in the region
            region_pixels = list(image.crop((left, upper, right, lower)).getdata())
            if 0 in region_pixels:
                max_color = 0
            else:
                max_color = 255
            # Assign the maximum color to the entire region
            for w in range(x):
                for h in range(x):
                    new_image.putpixel((left + w, upper + h), max_color)

    return new_image


def image_to_gray_matrix(image):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Get the dimensions of the image
    width, height = gray_image.size

    # Create an empty matrix to store pixel values
    gray_matrix = []

    # Iterate over each pixel in the image
    for y in range(height):
        row = []
        for x in range(width):
            # Get the grayscale value of the pixel
            pixel_value = gray_image.getpixel((x, y))
            row.append(pixel_value)
        gray_matrix.append(row)

    return gray_matrix

def find_wall_coord(mat):
    coordinates = []
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            if mat[i][j] == 0:
                coordinates.append((i, j))

    return coordinates

def find_rectangles(coords):
    from itertools import product

    # 将输入的坐标转换为集合，方便快速查询
    pixel_set = set(coords)

    # 用于记录已经处理过的像素
    visited = set()

    # 存储最终的矩形
    rectangles = []

    def is_valid_rectangle(x1, y1, x2, y2):
        # 检查矩形内部所有像素是否都在原始集合中
        for x, y in product(range(x1, x2 + 1), range(y1, y2 + 1)):
            if (x, y) not in pixel_set:
                return False
        return True

    def find_max_rectangle(x, y):
        max_x, max_y = x, y
        while True:
            found = False
            if is_valid_rectangle(x, y, max_x + 1, max_y):
                max_x += 1
                found = True
            if is_valid_rectangle(x, y, max_x, max_y + 1):
                max_y += 1
                found = True
            if not found:
                break
        return max_x, max_y

    for x, y in coords:
        if (x, y) not in visited:
            # 从当前像素出发尝试扩展矩形
            max_x, max_y = find_max_rectangle(x, y)
            # 标记这个矩形内的所有像素为已访问
            for i, j in product(range(x, max_x + 1), range(y, max_y + 1)):
                visited.add((i, j))
            rectangles.append(((x, y), (max_x, max_y)))

    return rectangles


def coord2yaml(coords):
    scale = 0.1
    walls = []
    for idx, coord in enumerate(coords):
        x0 = coord[0][0]
        y0 = coord[0][1]
        x1 = coord[1][0]
        y1 = coord[0][1]
        wall = {
            'name': f'wall_{idx}',
            'pos': [(x0+x1)/2*scale,(y0+y1)/2*scale, 0],
            'euler': [0, 0, 0],
            'type': 'box',
            'size': [abs((x1-x0))/2*scale,abs((y1-y0))/2*scale, 0.5],
            'group': 2,
            'rgba': [1.0, 1.0, 1.0, 0.5]
        }
        walls.append(wall)

    data = {
        'walls': {
            'name': 'walls',
            'pos': [0, 0, 0],
            'geoms': walls
        }
    }

    yaml_str = yaml.dump(data, default_flow_style=False, indent=None).replace('\n', ',\n').rstrip(',')

    return yaml_str
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def show_recs(rectangles):
    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 添加每个矩形到图上
    for (x1, y1), (x2, y2) in rectangles:
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(rect)

    # 设置坐标轴范围和比例
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_aspect('equal')

    # 显示图形
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()
def main():
    input_image_path = 'pngmap.png'
    output_image_path = 'processed_map.png'
    x = 5  # Size of the square region
    image = Image.open(input_image_path)
    # 去掉边框
    image = crop_gray_border(image)

    # 转为黑白双色像素图
    processed_image = max_pixel_value(image, x)

    # 转为矩阵
    mat = image_to_gray_matrix(processed_image)
    print(mat)
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    print(rows,cols,rows*cols)

    # 获取墙的坐标
    coord = find_wall_coord(mat)
    print(len(coord))

    # 转为矩形墙
    recs=find_rectangles(coord)
    # show_recs(recs)
    # print(len(recs))

    # 转为yaml格式
    yaml_output = coord2yaml(recs)
    with open("output.yaml", 'w') as file:
        file.write(yaml_output)
    processed_image.save(output_image_path)
    # processed_image.show()

if __name__ == "__main__":
    main()
