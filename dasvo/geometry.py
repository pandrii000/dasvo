"""
Geometric solvers for DASVO.
"""

import cv2
import numpy as np

from dasvo.settings import SETTINGS


class GeometryBackend:
    def estimate_relative_pose(self, pts1, pts2, K, depth1=None):
        raise NotImplementedError


class EssentialMatrixBackend(GeometryBackend):
    def __init__(self):
        self.min_correspondences = SETTINGS.backends.essential.min_correspondences
        self.ransac_prob = SETTINGS.backends.essential.ransac_probability
        self.ransac_threshold = SETTINGS.backends.essential.ransac_threshold_px

    def estimate_relative_pose(self, pts1, pts2, K, depth1=None):
        """
        Estimates scale-ambiguous relative pose using Essential Matrix + RANSAC.
        """
        if len(pts1) < self.min_correspondences:
            return None, None

        essential, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=self.ransac_prob, threshold=self.ransac_threshold
        )

        if essential is None:
            return None, None

        _, rotation, translation, _ = cv2.recoverPose(
            essential, pts1, pts2, K, mask=mask
        )

        return rotation, translation


class PnPBackend(GeometryBackend):
    def __init__(self):
        self.min_correspondences = SETTINGS.backends.pnp.min_correspondences
        self.min_inliers = SETTINGS.backends.pnp.min_inliers
        self.iterations_count = SETTINGS.backends.pnp.iterations_count
        self.reprojection_error = SETTINGS.backends.pnp.reprojection_error_px
        self.confidence = SETTINGS.backends.pnp.confidence
        self.max_translation_norm = SETTINGS.backends.pnp.max_translation_norm_m
        
        solver_str = SETTINGS.backends.pnp.solver
        if solver_str == "EPNP":
            self.solver = cv2.SOLVEPNP_EPNP
        elif solver_str == "ITERATIVE":
            self.solver = cv2.SOLVEPNP_ITERATIVE
        elif solver_str == "P3P":
            self.solver = cv2.SOLVEPNP_P3P
        elif solver_str == "AP3P":
            self.solver = cv2.SOLVEPNP_AP3P
        elif solver_str == "SQPNP":
            self.solver = cv2.SOLVEPNP_SQPNP
        else:
            self.solver = cv2.SOLVEPNP_EPNP

    def estimate_relative_pose(self, pts1, pts2, K, depth1=None):
        """
        Estimates metric relative pose using 3D-2D PnP + RANSAC.
        """
        if depth1 is None:
            raise ValueError("Depth map must be provided for PnP Backend.")

        if len(pts1) < self.min_correspondences:
            return None, None

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        pts3d = []
        valid_pts2d = []

        height, width = depth1.shape

        for i in range(len(pts1)):
            pt = pts1[i]
            # Handle potential shape differences (e.g. [[x, y]] vs [x, y])
            if len(pt.shape) > 1:
                pt = pt.flatten()
                
            u, v = int(pt[0]), int(pt[1])

            if u < 0 or u >= width or v < 0 or v >= height:
                continue

            z = depth1[v, u]

            if z <= SETTINGS.camera.depth_min_m or z > SETTINGS.camera.depth_max_m or np.isnan(z):
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            pts3d.append([x, y, z])
            valid_pts2d.append(pts2[i])

        pts3d = np.array(pts3d, dtype=np.float32)
        valid_pts2d = np.array(valid_pts2d, dtype=np.float32)

        if len(pts3d) < self.min_correspondences:
            return None, None

        success, rvec, translation, inliers = cv2.solvePnPRansac(
            pts3d,
            valid_pts2d,
            K,
            None,
            iterationsCount=self.iterations_count,
            reprojectionError=self.reprojection_error,
            confidence=self.confidence,
            flags=self.solver,
        )

        if not success:
            return None, None
        if inliers is None or len(inliers) < self.min_inliers:
            return None, None
        if np.linalg.norm(translation) > self.max_translation_norm:
            return None, None

        rotation, _ = cv2.Rodrigues(rvec)
        return rotation, translation
