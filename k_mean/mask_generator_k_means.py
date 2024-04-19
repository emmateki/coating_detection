import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

#  folder containing the images
folder_path = "path"

# output folder to save the masks
output_folder = "path"

num_clusters = 3

def k_mean_mask(folder_path,output_folder,num_clusters):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')): 
            image_path = os.path.join(folder_path, filename)
            
            image = cv2.imread(image_path)
            
            blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
            gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
            reshaped_image = gray_image.reshape((-1, 1))

            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(reshaped_image)

            labels = kmeans.labels_
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            # to determine which label belong to coating
            centroids = kmeans.cluster_centers_
            distances_to_upper_boundary = [centroid[0] for centroid in centroids]
            closest_to_upper_boundary = np.argmin(distances_to_upper_boundary)

            label_counts_sorted_indices = np.argsort(label_counts)

            smallest_cluster_label = unique_labels[label_counts_sorted_indices[0]]
            middle_cluster_label = unique_labels[label_counts_sorted_indices[1]]
            largest_cluster_label = unique_labels[label_counts_sorted_indices[2]]

            cluster_colors = {smallest_cluster_label: [0, 0, 0], largest_cluster_label: [0, 0, 0], middle_cluster_label: [0, 0, 0]}

            if closest_to_upper_boundary == middle_cluster_label:
                cluster_colors[smallest_cluster_label] = [255, 255, 255]
            else:
                cluster_colors[middle_cluster_label] = [255, 255, 255]

            segmented_img = np.zeros_like(image)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    label = labels[i * image.shape[1] + j]
                    segmented_img[i, j] = cluster_colors[label]

            #  polishing directly on the segmented image
            mask = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            touching_both_sides = False
            for i in range(1, num_labels):
                left_side = labels[:, 0] == i
                right_side = labels[:, -1] == i
                if np.any(left_side) and np.any(right_side):
                    touching_both_sides = True
                    break
            if touching_both_sides:
                largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                largest_component_mask = np.uint8(labels == largest_component_index) * 255
                filtered_mask = cv2.bitwise_and(thresh, largest_component_mask)
            else:
                filtered_mask = mask
            filtered_mask = cv2.cvtColor(filtered_mask, cv2.IMREAD_GRAYSCALE)
            # Convert the filtered mask to binary
            _, binary_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)
            binary_mask_path = os.path.join(output_folder, f"{filename}")
            cv2.imwrite(binary_mask_path, binary_mask)

k_mean_mask(folder_path,output_folder,num_clusters)
