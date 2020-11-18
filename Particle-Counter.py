"""
Particle_Counter.py
"""
import cv2
import pylab
import matplotlib.pyplot as plt
from scipy import ndimage


def particle_counter(img_path):
    count_values = list()
    print("Reading image: " + img_path)
    image_name = img_path.split("\\")[-1]
    max_value = 255
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    threshold_type = cv2.THRESH_BINARY
    block_size = 5
    c = -1
    with open(image_name.split(".")[0] + ".txt", "w") as file:
        im = cv2.imread(img_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        im_threshold = cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, c)
        label_array, particle_count = ndimage.measurements.label(im_threshold)
        count_values.append(particle_count)
        file.write("Reading image: " + img_path + "\n")
        file.write(str(particle_count) + "\n")
        pylab.figure(1)
        pylab.imshow(im_threshold)
        pylab.savefig(img_path.split(".")[0] + "_Output.png")
    return count_values


if __name__ == '__main__':
    print(particle_counter("C:\\cell\\digital_assay\\sample.jpg"))
