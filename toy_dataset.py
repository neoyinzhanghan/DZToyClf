import numpy as np
from PIL import Image

def create_rectangle(rectangle_size, class_id, downsample_rate=16):
    img = np.zeros((rectangle_size, rectangle_size, 3), np.uint8)  # Create a small rectangle

    img_small = np.zeros((rectangle_size//downsample_rate, rectangle_size//downsample_rate, 3), np.uint8)  # Create a small rectangle
    
    # Random color for the rectangle
    color = np.random.randint(0, 255, 3)
    img[:, :] = color
    img_small[:, :] = color
    
    if class_id == 0:
        # Add a small plus sign in the middle of the rectangle
        plus_size = 10
        center_x, center_y = rectangle_size // 2, rectangle_size // 2
        img[center_y - plus_size // 2 : center_y + plus_size // 2, center_x - 1 : center_x + 1] = [255, 255, 255]  # vertical line
        img[center_y - 1 : center_y + 1, center_x - plus_size // 2 : center_x + plus_size // 2] = [255, 255, 255]  # horizontal line
    
    elif class_id == 1:
        # Add a small circle in the middle of the rectangle
        circle_radius = 5
        center_x, center_y = rectangle_size // 2, rectangle_size // 2
        for i in range(rectangle_size):
            for j in range(rectangle_size):
                if (i - center_x) ** 2 + (j - center_y) ** 2 < circle_radius**2:
                    img[j, i] = [255, 255, 255]  # white circle
    
    else:
        raise ValueError("Invalid class_id")
    
    return img, img_small

def generate_data(class_id):

    # Parameters
    rectangle_size = 224
    downsample_rate = 16
    width = 224 * 16
    height = 224 * 16


    img = np.zeros((height, width, 3), np.uint8)
    img_small = np.zeros((height // downsample_rate, width // downsample_rate, 3), np.uint8)

    rectangle, rectangle_small = create_rectangle(rectangle_size, class_id) 

    x = np.random.randint(0, width - rectangle_size)
    y = np.random.randint(0, height - rectangle_size)

    x_downsampled = x // downsample_rate
    y_downsampled = y // downsample_rate

    img[y : y + rectangle_size, x : x + rectangle_size] = rectangle
    img_small[y_downsampled : y_downsampled + rectangle_size // downsample_rate, x_downsampled : x_downsampled + rectangle_size // downsample_rate] = rectangle_small

    img = Image.fromarray(img)
    img_small = Image.fromarray(img_small)

    return img, img_small, class_id

if __name__ == "__main__":

    img, img_small, class_id = generate_data(0)
    # save the images at rectangle_0.jpg and rectangle_0_downsampled.jpg

    img.save("rectangle_0.png")
    img_small.save("rectangle_0_downsampled.png")

    img, img_small, class_id = generate_data(1)
    # save the images at rectangle_1.jpg and rectangle_1_downsampled.jpg

    img.save("rectangle_1.png")
    img_small.save("rectangle_1_downsampled.png")
