import cv2
import numpy as np

from dasvo.settings import SETTINGS


class FrontEnd:
    def process_frame(self, img):
        raise NotImplementedError


class KLTFrontEnd(FrontEnd):
    def __init__(self):
        self.max_corners = SETTINGS.frontends.klt.max_corners
        self.quality_level = SETTINGS.frontends.klt.quality_level
        self.min_distance = SETTINGS.frontends.klt.min_distance

        self.lk_params = {
            "winSize": tuple(SETTINGS.frontends.klt.win_size),
            "maxLevel": SETTINGS.frontends.klt.max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                SETTINGS.frontends.klt.criteria_count,
                SETTINGS.frontends.klt.criteria_eps,
            ),
        }

        self.prev_gray = None
        self.prev_pts = None

    def process_frame(self, img):
        """
        Returns matched point coordinates as (pts_prev, pts_curr).
        """
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < SETTINGS.frontends.klt.redetect_below:
            self.prev_pts = cv2.goodFeaturesToTrack(
                curr_gray,
                mask=None,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                useHarrisDetector=SETTINGS.frontends.klt.use_harris_detector,
                k=SETTINGS.frontends.klt.harris_k,
            )
            self.prev_gray = curr_gray
            return None, None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params
        )

        if status is not None:
            status = status.flatten()
            good_old = self.prev_pts[status == 1]
            good_new = curr_pts[status == 1]
        else:
            good_old = np.array([])
            good_new = np.array([])

        self.prev_gray = curr_gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            curr_gray,
            mask=None,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            useHarrisDetector=SETTINGS.frontends.klt.use_harris_detector,
            k=SETTINGS.frontends.klt.harris_k,
        )

        return good_old, good_new


class ORBFrontEnd(FrontEnd):
    def __init__(self):
        self.orb = cv2.ORB_create(
            nfeatures=SETTINGS.frontends.orb.nfeatures,
            scaleFactor=SETTINGS.frontends.orb.scale_factor,
            nlevels=SETTINGS.frontends.orb.nlevels,
            edgeThreshold=SETTINGS.frontends.orb.edge_threshold,
            firstLevel=SETTINGS.frontends.orb.first_level,
            WTA_K=SETTINGS.frontends.orb.wta_k,
            scoreType=cv2.ORB_HARRIS_SCORE if SETTINGS.frontends.orb.score_type == "HARRIS_SCORE" else cv2.ORB_FAST_SCORE,
            patchSize=SETTINGS.frontends.orb.patch_size,
            fastThreshold=SETTINGS.frontends.orb.fast_threshold,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=SETTINGS.frontends.orb.cross_check)
        self.ratio_test = SETTINGS.frontends.orb.ratio_test
        self.min_matches = SETTINGS.frontends.orb.min_matches
        self.prev_kps = None
        self.prev_des = None

    def process_frame(self, img):
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        curr_kps, curr_des = self.orb.detectAndCompute(curr_gray, None)

        if self.prev_des is None or curr_des is None or len(curr_kps) < 2 or len(self.prev_kps) < 2:
            self.prev_kps = curr_kps
            self.prev_des = curr_des
            return None, None

        # Ensure descriptors are of type CV_8U for NORM_HAMMING
        if self.prev_des.dtype != np.uint8:
            self.prev_des = self.prev_des.astype(np.uint8)
        if curr_des.dtype != np.uint8:
            curr_des = curr_des.astype(np.uint8)

        if not SETTINGS.frontends.orb.cross_check:
            matches = self.matcher.knnMatch(self.prev_des, curr_des, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    match, neighbor = m_n
                    if match.distance < self.ratio_test * neighbor.distance:
                        good_matches.append(match)
                elif len(m_n) == 1:
                    good_matches.append(m_n[0])
        else:
            matches = self.matcher.match(self.prev_des, curr_des)
            good_matches = matches

        if len(good_matches) < self.min_matches:
            self.prev_kps = curr_kps
            self.prev_des = curr_des
            return None, None

        pts_prev = np.float32([self.prev_kps[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([curr_kps[m.trainIdx].pt for m in good_matches])

        self.prev_kps = curr_kps
        self.prev_des = curr_des

        return pts_prev, pts_curr
