import numpy as np
from PIL import Image

def create_rectangle(rectangle_size, class_id):
    img = np.zeros((rectangle_size, rectangle_size, 3), np.uint8)  # Create a small rectangle
    
    # Random color for the rectangle
    color = np.random.randint(0, 255, 3)
    img[:, :] = color
    
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
    
    return img

def generate_data(class_id):

    # Parameters
    rectangle_size = 224
    downsample_rate = 16
    width = 224 * 16
    height = 224 * 16


    img = np.zeros((height, width, 3), np.uint8)

    rectangle = create_rectangle(rectangle_size, class_id) 

    x = np.random.randint(0, width - rectangle_size)
    y = np.random.randint(0, height - rectangle_size)

    img[y : y + rectangle_size, x : x + rectangle_size] = rectangle

    img = Image.fromarray(img)

    downsampled_img = img.resize((width // downsample_rate, height // downsample_rate))

    return img, downsampled_img, class_id

if __name__ == "__main__":

    # Parameters
    rectangle_size = 224
    downsample_rate = 16
    width = 224 * 16
    height = 224 * 16

    # Create a blank image of size width x height
    img0 = np.zeros((height, width, 3), np.uint8)

    # Generate a rectangle with class_id = 0 or 1
    rectangle = create_rectangle(rectangle_size, class_id=0) 

    # Randomly place the rectangle in the large image
    x = np.random.randint(0, width - rectangle_size)
    y = np.random.randint(0, height - rectangle_size)

    # Place the rectangle in the main image
    img0[y : y + rectangle_size, x : x + rectangle_size] = rectangle

    # Save the image
    img0 = Image.fromarray(img0)
    img0.save("rectangle_0.jpg")

    # now try class_id=1

    # Create a blank image of size width x height
    img1 = np.zeros((height, width, 3), np.uint8)

    # Generate a rectangle with class_id = 0 or 1
    rectangle = create_rectangle(rectangle_size, class_id=1) 

    # Randomly place the rectangle in the large image
    x = np.random.randint(0, width - rectangle_size)
    y = np.random.randint(0, height - rectangle_size)

    # Place the rectangle in the main image
    img1[y : y + rectangle_size, x : x + rectangle_size] = rectangle

    # Save the image
    img1 = Image.fromarray(img1)
    img1.save("rectangle_1.jpg")

    # now downsample img0 and img1 by a factor of 16 and save them as well
    img0_downsampled = img0.resize((width // downsample_rate, height // downsample_rate))
    img0_downsampled.save("rectangle_0_downsampled.jpg")

    img1_downsampled = img1.resize((width // downsample_rate, height // downsample_rate))
    img1_downsampled.save("rectangle_1_downsampled.jpg")

