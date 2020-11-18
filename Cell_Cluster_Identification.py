"""
Cell_Cluster_Identification.py
@author: Samhitha Kolla (samhitha.kolla@wustl.edu)
"""
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import math
import warnings

warnings.filterwarnings("ignore")


def findCluster(path, bound_img_path):
    """
    This function identifies the cluster from the image
    :param path: Image path
    :param bound_img_path: Image path consisting bounding boxes
    :return: counts, bounding boxes created
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(img, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Creating a structuring element to perform morphological operations
    k_size = (3, 3)
    kernelMorph = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)

    # Performing opening on the image
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelMorph, iterations=1)

    # Performing dilation on the image
    ksizeKernelDilate = (50, 1)
    dilated = cv2.dilate(morph, ksizeKernelDilate, iterations= 5)

    # Identifying contours in the image
    cnts = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    clr1, clr2, clr3 = 225, 255, 0,
    b = 10

    bounding_box_created = False
    if len(cnts) != 0:
        contour_rect_box = []
        conimage = []
        for c in cnts:
            # Identifying bounding box dimensions for each contour
            x, y, w, h = cv2.boundingRect(c)
            # Enclosing the contours in a rectangle
            if (w > 10 and h > 10 and w < img.shape[0] and h < img.shape[1]):
                conimage = cv2.rectangle(image, (x - b, y - b), (x - b + w + 2 * b, y - b + h + 2 * b),
                                         (clr1, clr2, clr3), -1)
                contour_rect_box.append((x, y, x + w, y + h, w, h))

        # Check whether the bounding box is created or not
        # If bounding box is created, then update the image with the identified bounding box,
        # Else print no clusters have been identified
        if len(conimage) > 0:
            cv2.imwrite(bound_img_path, conimage)
            bounding_box_created = True
        else:
            print("No clusters identified")
    return cnts, bounding_box_created



def count_points(roi):
    """
    This function counts the number of points(secretions) within the clusters
    :param roi: Region of Interest
    :return: labels, unique label counts
    """
    # Performing Mean Shift Filtering
    shifted = cv2.pyrMeanShiftFiltering(roi, 21, 51)

    # Converting the image to grayscale
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    # Thresholding using Binary and OTSU
    thrsh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Using Watershed Algorithm
    D = ndimage.distance_transform_edt(thrsh)
    localMax = peak_local_max(D, indices=False, min_distance=1, labels=thrsh)
    markers = ndimage.label(localMax)[0]
    lbls = watershed(-D, markers, mask=thrsh)
    
    return lbls, len(np.unique(lbls)) - 1


if __name__ == '__main__':
    # Image path
    path_main = "C:\\cell\\CellClustering\\LPS\\KO_LPS_1.jpg"
    filename = path_main.split(".")[0]
    with open(filename + "_Output.txt", "w") as file:
        file.write("Image Captured: " + str(path_main) + "\n")

        # change rotation for dilation

        rotation = 3
        bound_img_path = filename + '_BoundingBox.jpg'
        for i in range(0, rotation):
            # Checking whether clusters are identified in the image or not
            if i == 0:
                cnts,bounding_box_created = findCluster(path_main, bound_img_path)
                print(path_main)
                # When no clusters are identified, there is no bounding box
                if not bounding_box_created:
                    break
            else:
                cnts,bounding_box_created = findCluster(bound_img_path, bound_img_path)

        size = 80   # Change the size according to the images
        ogimage = cv2.imread(path_main, cv2.IMREAD_COLOR)

        no_of_clusters = 0
        for (i, c) in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if size < w < ogimage.shape[0] and size < h < ogimage.shape[1]:
                no_of_clusters += 1
                ROI = ogimage[y:y + h, x:x + w]

                # Counting the number of secretions within each cluster
                lbls, count = count_points(ROI)

                # Enclosing each cluster inside a circle
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(ogimage, (int(cX), int(cY)), int(radius), (255, 255, 255), 2)  # 5
                cv2.putText(ogimage, "#{}".format(count), (int(cX), int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 1.85,
                            (0, 255, 255), 3)

                file.write(str(count) + "\n")
                cv2.imwrite(filename + '_Output.jpg', ogimage)
        file.write("Clusters: " + str(no_of_clusters) + "\n")

        # To store the coordinates of bounding boxes
        # contoursdf = pd.DataFrame(contour_rect_box, columns=["x1", "y1", "x2", "y2", "w", "h"])
        # contoursdf.to_csv(path_main + str(image_counter) + " cells.csv", index=False)
