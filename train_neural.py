from tensorflow.keras import datasets
from neural import Model
import numpy as np
from PIL import Image

def resize_image(image, target_size=(20, 20)):
    if not np.any(image):
        return None
    img_array = np.array(image)

    binary_img = img_array > 100

    # Find bounding box of the digit
    coords = np.argwhere(binary_img)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the digit
    cropped_digit = img_array[y_min:y_max+1, x_min:x_max+1]

    digit_pil = Image.fromarray(cropped_digit)
    digit_resized = digit_pil.resize((20, 20), Image.LANCZOS)

    # Convert back to array and visualize
    final_image = np.array(digit_resized)

    new_image = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    new_image[y_offset:y_offset+20, x_offset:x_offset+20] = final_image
    normalized = new_image // 1
    return normalized

def clean_img(img):
    new = np.copy(img)
    canny = np.where(img < 128, 0, 255).astype(np.uint8)
    r_i = np.any(canny == 0, axis=1)  # If all rows are dark then it is noise
    c_i = np.any(canny == 0, axis=0)

    new[np.where(r_i == False)[0], :] = 0  # remove the noise
    new[:, np.where(c_i == False)[0]] = 0
    return new

(x, y), (test_images, test_labels) = datasets.mnist.load_data()
x = x[y != 0]
y = y[y != 0]
test_images = test_images[test_labels != 0]
test_labels = test_labels[test_labels != 0]
y[y == 9] = 0 # we will reperesent 9 as 0
test_labels[test_labels == 9] = 0

xx = []
for i in x:
    i = clean_img(i)
    i[np.where(i < 160)] = 0
    i = resize_image(i)
    xx.append(i)

x = np.array(xx)
x = x.reshape(len(y), 784)
test_images = test_images.reshape(len(test_labels), 784)

model = Model(folder_path="parameters2", use_existing=False)
model.gradientDesent(x, y, 5000, 1)

model.forward(test_images, test_labels)
model.predict()
model.print_model(test_labels, 0)
