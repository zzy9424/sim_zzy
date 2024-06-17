from PIL import Image


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

    new_image = Image.new(image.mode, (width, height))  # Create new image with the same mode

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

def main():
    input_image_path = 'pngmap.png'
    output_image_path = 'processed_map.png'
    x = 5  # Size of the square region

    # Open the image
    image = Image.open(input_image_path)
    image = crop_gray_border(image)
    image.show()
    # Process the image to get maximum value regions
    processed_image = max_pixel_value(image, x)
    mat = image_to_gray_matrix(processed_image)
    print(mat)
    # Save the processed image
    processed_image.save(output_image_path)
    processed_image.show()


if __name__ == "__main__":
    main()
