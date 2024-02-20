import numpy as np


class Plane:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_z(self, x, y):
        return (self.d - self.a * x - self.b * y) / self.c

    def get_line_intersection(self, start_point: np.ndarray, direction: np.ndarray):
        v = np.array([self.a, self.b, self.c])
        # t = (self.d - np.dot(v, start_point)) / np.dot(v, direction)
        t = self.d / np.dot(v, direction)
        return t * direction, t


table_plane = Plane(0.1, 0, 0.99, .05476)


def compute_affine_transform(P, Q):
    """
    Compute the affine transformation matrix A and translation vector t
    to map points P to points Q.

    Args:
    - P: Array of shape (n, 2) representing the vertices of polygon P
    - Q: Array of shape (n, 2) representing the vertices of polygon Q

    Returns:
    - A: Affine transformation matrix of shape (2, 2)
    - t: Translation vector of shape (2,)
    """
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Compute translation vector
    t = centroid_Q - centroid_P

    # Compute covariance matrix
    H = (P - centroid_P).T @ (Q - centroid_Q)

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Compute scaling matrix
    S = np.eye(2)
    S[1, 1] = np.linalg.det(R)
    # S[0, 0] = np.linalg.det(R)

    # Compute affine transformation matrix
    A = R @ S

    return A, t


class RobosuiteTransforms:

    def __init__(self, extrinsic_mat, intrinsic_mat):
        self.extrinsic_mat = extrinsic_mat
        self.intrinsic_mat = intrinsic_mat

        # Calculate inverse matrices
        self.intrinsic_inv = np.linalg.inv(self.intrinsic_mat)
        self.extrinsic_inv = np.linalg.inv(self.extrinsic_mat)

    # Given XY coordinates relative to the camera, provide the expected Z coordinate
    def get_relative_coord(self, pixel_coord):
        # print(self.intrinsic_inv.shape, pixel_coord)
        relative_coord = np.matmul(self.intrinsic_inv, pixel_coord)
        relative_coord = np.append(relative_coord, 1)
        return relative_coord

    # Given XY coordinates relative to the camera, provide the XYZ coordinates relative to the camera
    def get_world_coord(self, relative_camera_coord, solve_for_z=False):
        world_coord = np.matmul(self.extrinsic_inv, relative_camera_coord)
        if solve_for_z:
            start_point = self.extrinsic_mat[:3, 3]
            direction = world_coord[:3]
            _, scale = table_plane.get_line_intersection(start_point, direction)
            scale = table_plane.get_z(*world_coord[:2]) / world_coord[2]
            world_coord *= scale
        return world_coord

    def pixel_to_world_coord(self, pixel_coord, solve_for_z=False):
        relative_coord = self.get_relative_coord(pixel_coord)
        world_coord = self.get_world_coord(relative_coord, solve_for_z=solve_for_z)
        return world_coord