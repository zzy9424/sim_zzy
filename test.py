def find_rectangles(pixels):
    from itertools import product

    # 将输入的坐标转换为集合，方便快速查询
    pixel_set = set(pixels)

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

    for x, y in pixels:
        if (x, y) not in visited:
            # 从当前像素出发尝试扩展矩形
            max_x, max_y = find_max_rectangle(x, y)
            # 标记这个矩形内的所有像素为已访问
            for i, j in product(range(x, max_x + 1), range(y, max_y + 1)):
                visited.add((i, j))
            rectangles.append(((x, y), (max_x, max_y)))

    return rectangles

# 示例
pixels = [(1,1),(1,2),(2,1),(3,3),(3,4)]
rectangles = find_rectangles(pixels)
print(rectangles)