import open3d
import numpy as np
import random


# Given points classified as driveable, estimate driveable plane
# Points should be provided as a list of [x, y, z] coordinates
# https://github.com/qiaoxu123/Self-Driving-Cars/blob/master/Part3-Visual_Perception_for_Self-Driving_Cars/Module5-Semantic_Segmentation/Module5-Semantic_Segmentation.md
# ax + by + z = d, p is parameters [a, b, d]
def ransac(points, threshold=0.2*len(points)):
    while True:
        # sample 3 points
        three_points = random.choices(points, k=3)
        points_matrix = np.asarray(three_points)
        # Determine plane
        A = np.asarray([[point[0], point[1], -1] for point in three_points])
        B = -1 * points_matrix[:, 2]
        p = np.linalg.inv(np.transpose(A) * A) * np.transpose(A) * B
        # Find number of points in plane
        # Break if over threshold
    # Recompute using all inliers


def rgbd_to_3d(image):
    pcd = open3d.geometry.create_point_cloud_from_rgbd_image(image, open3d.camera.pinhole_camera_intrinsic)
    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])