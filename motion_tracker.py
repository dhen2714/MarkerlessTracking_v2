import pandas as pd
import cv2
import numpy as np
from camera_view import CameraView
import sys
import os
import time


def load_stereo_views(stereo_calib_file='Stereo_calibration.npz'):
    """
    Reads a stereo calibration npz file and returns two CameraView objects which
    can be loaded into StereoFeatureTracker.
    """
    f = np.load(stereo_calib_file)
    K1 = f['K1']
    K2 = f['K2']
    dc1 = np.squeeze(f['dc1'])
    dc2 = np.squeeze(f['dc2'])
    R = f['R']
    t = f['T']
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K2, np.hstack((R, t.reshape(3, 1))))
    fc1 = np.array([K1[0, 0], K1[1, 1]])
    fc2 = np.array([K2[0, 0], K2[1, 1]])
    pp1 = np.array([K1[0, 2], K1[1, 2]])
    pp2 = np.array([K2[0, 2], K2[1, 2]])
    kk1 = np.array([dc1[0], dc1[1], dc1[4]])
    kk2 = np.array([dc2[0], dc2[1], dc2[4]])
    kp1 = np.array([dc1[2], dc1[3]])
    kp2 = np.array([dc2[2], dc2[3]])
    view1 = CameraView(P1, fc1, pp1, kk1, kp1)
    view2 = CameraView(P2, fc2, pp2, kk2, kp2)
    return view1, view2


def mdot(*args):
    """Matrix multiplication for more than 2 arrays."""
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret


def mat2vec(H):
    """
    Converts a 4x4 representation of pose to a 6 vector.
    Inputs:
        H - 4x4 matrix.
    Outputs:
        v - [yaw,pitch,roll,x,y,z] (yaw,pitch,roll are in degrees)
    """
    sy = -H[2, 0]
    cy = 1-(sy*sy)

    if cy > 0.00001:
        cy = np.sqrt(cy)
        cx = H[2, 2]/cy
        sx = H[2, 1]/cy
        cz = H[0, 0]/cy
        sz = H[1, 0]/cy
    else:
        cy = 0.0
        cx = H[1, 1]
        sx = -H[1, 2]
        cz = 1.0
        sz = 0.0

    r2deg = (180/np.pi)
    return np.array([np.arctan2(sx, cx)*r2deg, np.arctan2(sy, cy)*r2deg,
                     np.arctan2(sz, cz)*r2deg,
                     H[0, 3], H[1, 3], H[2, 3]])


def vec2mat(*args):
    """
    Converts a six vector represenation of motion to a 4x4 matrix.
    Assumes yaw, pitch, roll are in degrees.
    Inputs:
        *args - either 6 numbers (yaw,pitch,roll,x,y,z) or an array with
             6 elements.
     Outputs:
        t     - 4x4 matrix representation of six vector.
    """
    if len(args) == 6:
        yaw = args[0]
        pitch = args[1]
        roll = args[2]
        x = args[3]
        y = args[4]
        z = args[5]
    elif len(args) == 1:
        yaw = args[0][0]
        pitch = args[0][1]
        roll = args[0][2]
        x = args[0][3]
        y = args[0][4]
        z = args[0][5]

    ax = np.radians(yaw)
    ay = np.radians(pitch)
    az = np.radians(roll)
    t1 = np.array([[1, 0, 0, 0],
                   [0, np.cos(ax), -np.sin(ax), 0],
                   [0, np.sin(ax), np.cos(ax), 0],
                   [0, 0, 0, 1]])
    t2 = np.array([[np.cos(ay), 0, np.sin(ay), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ay), 0, np.cos(ay), 0],
                   [0, 0, 0, 1]])
    t3 = np.array([[np.cos(az), -np.sin(az), 0, 0],
                   [np.sin(az), np.cos(az), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    tr = np.array([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return mdot(tr, t3, t2, t1)


def detect_outliers(array):
    """
    Removes outliers based on modified Z-score. Translated from Andre's IDL
    code.
    Inputs:
        array     - 1D array
    Outputs:
        outliers  - Array of indicies corresponding to outliers
    """
    med = np.median(array)
    MAD = np.median(np.abs(array - med))

    if MAD == 0:
        meanAD = np.mean(np.abs(array - med))
        if meanAD == 0:
            return np.array([])
        zscore = np.abs(array - med)/(1.253314*meanAD)
    else:
        zscore = np.abs(array - med)/(1.4826*MAD)
    return np.where(zscore > 3.5)[0]


class LandmarkDatabase:
    """
    Database of 3D landmark positions and their descriptors.
    Attributes:
        landmarks     - 3D position of landmarks
        descriptors   - landmark feature vectors
        miss_counts   - number of consecutive frames in which landmark not seen
        miss_thresh   - if miss_count > miss_thresh, landmark is pruned
    """

    def __init__(self, X, descriptors, miss_thresh=None):
        self.landmarks = X
        self.descriptors = descriptors
        self.miss_counts = np.zeros(len(X), dtype=int)
        self.miss_thresh = miss_thresh

    def __len__(self):
        return len(self.landmarks)

    def update(self, X, descriptors, landmarks_used=None):
        if len(self) == 0:
            self.landmarks = X
            self.descriptors = descriptors
            self.miss_counts = np.zeros(len(X), dtype=int)
        else:
            self.landmarks = np.vstack((self.landmarks, X))
            self.descriptors = np.vstack((self.descriptors, descriptors))
            self.miss_counts = np.concatenate((self.miss_counts,
                                               np.zeros(len(X))))

    def trim(self, indices):
        """Removes entries with given indices from database"""
        self.landmarks = np.delete(self.landmarks, indices, axis=0)
        self.descriptors = np.delete(self.descriptors, indices, axis=0)
        self.miss_counts = np.delete(self.miss_counts, indices)

    def update_landmark_usage(self, landmark_indices):
        """
        Updates miss counts for entries in database. If indices are in the input
        array landmark_indices, then their corresponding miss count is set to 0.
        Otherwise, miss count is incremented by 1.

        Database entries with miss counts greater than miss threshold are then
        removed.
        """
        mask_used = np.zeros(len(self), dtype=bool)
        mask_unused = np.ones(len(self), dtype=bool)
        mask_used[landmark_indices] = True
        mask_unused[landmark_indices] = False

        self.miss_counts[mask_used] = 0
        self.miss_counts[mask_unused] += 1
        if self.miss_thresh is not None:
            self.trim(np.where(self.miss_counts >= self.miss_thresh)[0])

    def save(self, output_filename):
        """
        Saves database to pickle format.
        """
        raw_data = [[X, m, desc] for X, m, desc in zip(
                    self.landmarks, self.miss_counts, self.descriptors)]
        pd.DataFrame(data=raw_data,
                     columns=['Position',
                              'Miss count',
                              'Descriptor']).to_pickle(output_filename)

class StereoFeatureTracker:

    def __init__(self, view1, view2):
        assert all([isinstance(view, CameraView) for view in (view1, view2)]), \
            'Arguments must be CameraView objects!'
        self.view1 = view1
        self.view2 = view2
        self.database = LandmarkDatabase(np.array([]), np.array([]))
        self.currentPose = np.zeros(6, dtype=np.float64)  # rX, rY, rZ, X, Y, Z
        self.frameNumber = 0

        self.matches_inframe = []  # Could be useful for diagnostic purposes
        self.matches_db_view1 = []
        self.matches_db_view2 = []
        self.metadata = []  # Timing data, size of database, etc.
        self.pose_history = [] # Past poses with flag

        # Attributes below are 'public'

        # est_method can be set to either 'gn' or 'ls'. 'ls' uses Horn's method
        # to estimate pose, whereas 'gn' uses least squares minimisation with
        # iterations of the Gauss-Newton method
        self.est_method = 'ls'
        self.ratioTest = True  # Apply ratio test in descriptor matching
        self.distRatio = 0.6  # Distance ratio cutoff for ratio test
        self.binaryMatch = False  # Use Hamming norm instead of L2
        self.matcher = cv2.BFMatcher()  # Initialize match handler
        # If estimated pose is too different from previous pose, reject pose est
        self.pose_threshold = np.array([10, 10, 10, 20, 20, 20])

        self.filtering = False
        self.current_state = np.zeros(12, dtype=np.float64) # includes velocity
        self.dt = 10e-3
        self.state_transition = np.eye(12)
        self.state_transition[:6, 6:] = np.eye(6)
        self.observation_model = np.zeros((6, 12))
        self.observation_model[:6, :6] = np.eye(6)

        self.state_covariance = np.zeros((12, 12))
        self.pose_covariance = np.zeros((12, 12))
        self.process_covariance = 1e2*np.eye(12)

        # Used to calculate a running mean and variance of pose.
        self.aggregate = (1, np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64))

        # Compute rectifying transforms.
        try:
            Prec1, Prec2, self.Tr1, self.Tr2 = self.rectify_fusiello()
            DD1, DD2 = self.generate_dds(self.Tr1, self.Tr2)
            Prec1, Prec2, self.Tr1, self.Tr2 = self.rectify_fusiello(DD1, DD2)
            print('Rectifying transforms successfully calculated.')
        except TypeError:
            print('Projection matrices could not be rectified! Re-check that ' +
                  'they have been entered correctly!')

        print('StereoFeatureTracker initialized.\n')

    def set_detectors(self, detector):
        self.view1.set_detAndDes(detector)
        self.view2.set_detAndDes(detector)

    def process_frame(self, image1, image2, save_poses=True, verbose=True):
        """
        Method to process a new frame, calculating a new pose estimate. If
        save_poses is True, poses are saved to self.pose_history.
        Inputs:
            image1, image2 - Images from camera view 1 and view 2 respectively.
        Outputs:
            New pose estimate in [rX rY rZ X Y Z] form.
            flag - 0 if pose estimated successfully, 1 otherwise.
        """
        flag = 0  # Returns 0 if pose estimated successfully

        # Suppress text output if verbose=False
        if not verbose:
            sys.stdout = open(os.devnull, 'w')
        processFrameStart = time.perf_counter()

        # Update camera views, find keypoints in updated views.
        loadTime1, detTime1, correctDistTime1 = self.view1.process_frame(image1)
        loadTime2, detTime2, correctDistTime2 = self.view2.process_frame(image2)

        # Perform intra-frame matching, get indices of matched keypoints.
        matchFrameStart = time.perf_counter()
        matches_inframe = self.match_descriptors(self.view1.descriptors,
                                                 self.view2.descriptors,
                                                 matching_type='intra_frame')
        # Remove keypoints in view 1 that have been matched to multiple
        # keypoints in view 2.
        self.matches_inframe = self.remove_duplicate_matches(matches_inframe)
        in1, in2 = self.extract_match_indices(self.matches_inframe)
        # Apply epipolar constraint to intra-frame matches.
        inEpi = self.epipolar_constraint(self.view1.key_coords[in1],
                                         self.view2.key_coords[in2],
                                         self.Tr1, self.Tr2)
        in1 = in1[inEpi]
        in2 = in2[inEpi]
        matchFrameTime = time.perf_counter() - matchFrameStart

        # frameDescriptors is an array of descriptors for intra-frame matches
        self.view1.descriptors[in1] = (self.view1.descriptors[in1] +
                                       self.view2.descriptors[in2])/2
        self.view2.descriptors[in2] = (self.view1.descriptors[in1] +
                                       self.view2.descriptors[in2])/2
        frameDescriptors = self.view1.descriptors[in1]

        # Triangulate intra-frame matched keypoints.
        X = self.triangulate_keypoints(self.view1.P, self.view2.P,
                                       self.view1.key_coords[in1],
                                       self.view2.key_coords[in2])

        # Estimate current pose
        if self.frameNumber == 0:

            self.database.update(X, frameDescriptors)
            matchDBTime, poseEstTime, used_landmarks1, used_landmarks2 \
                = 0, 0, np.array([]), np.array([])

        # Estimate pose using Horn's method, finding the required transformation
        # between triangulated points in current frame and matched landmarks in
        # the database.
        elif self.est_method == 'ls':

            matchDBTime, poseEstTime, used_landmarks, flag = \
                self.estimate_pose_ls(X, frameDescriptors)
            used_landmarks1 = used_landmarks
            used_landmarks2 = used_landmarks

        # Estimate pose using Gauss-Newton iterations, finding required
        # transformation that minimizes pixel projection error between keypoints
        # and matched landmarks in database.
        elif self.est_method == 'gn':

            matchDBTime, poseEstTime, used_landmarks1, used_landmarks2, flag = \
                self.estimate_pose_gn(X, frameDescriptors, in1, in2)

        # Update landmark usage, prune landmarks that have not bee used for too
        # many consecutive frames.
        used_landmark_indices = np.union1d(used_landmarks1, used_landmarks2)
        self.database.update_landmark_usage(used_landmark_indices.astype(int))

        # Running variance and mean used for weighted least squares estimation
        # in GN estimation.
        self.update_aggregate()

        processFrameTime = time.perf_counter() - processFrameStart

        n_used_landmarks1 = len(used_landmarks1)
        n_used_landmarks2 = len(used_landmarks2)

        print('\nPose estimate for frame {} is:\n {} \n'.format(
            self.frameNumber, self.currentPose))

        print('{} landmarks in database.\n'.format(len(self.database)))

        # Save pose history
        if save_poses:
            self.pose_history.append([self.frameNumber, *self.currentPose,
                                      flag])

        # Save metadata for current frame.
        frame_metadata = (self.frameNumber,
                          loadTime1, loadTime2,
                          detTime1, detTime2,
                          correctDistTime1, correctDistTime2,
                          matchFrameTime,
                          matchDBTime,
                          poseEstTime,
                          processFrameTime,
                          self.view1.n_keys, self.view2.n_keys,
                          len(in1),
                          n_used_landmarks1, n_used_landmarks2,
                          len(self.database))

        self.metadata.append(frame_metadata)
        self.frameNumber += 1  # Update frame number.

        # Return text output to normal if verbosity was set to false.
        if not verbose:
            sys.stdout = sys.__stdout__
        return self.currentPose, flag

    def save_poses(self, filePath):
        """
        Saves poses with flag and frame number in pickle format.
        """
        pd.DataFrame(data=np.array(self.pose_history)[:, 1:],
                     columns=['rX', 'rY', 'rZ', 'X', 'Y', 'Z', 'Flag'],
                     index=np.array(self.pose_history)[:, 0]).to_pickle(filePath)

    def save_metadata(self, filePath):
        """
        Saves metadata as a pandas dataframe, serialized in pickle format.
        """
        pd.DataFrame(data=np.array(self.metadata)[:, 1:],
                     columns=['View1 load(s)', 'View2 load(s)',
                              'View1 det/des(s)', 'View2 det/des(s)',
                              'View1 distcorr(s)', 'View2 distcorr(s)',
                              'Frame match(s)',
                              'Database match(s)',
                              'Pose est(s)',
                              'Frame process(s)',
                              '#View1 keypoints', '#View2 keypoints',
                              '#Intra-frame matches',
                              '#View1 keypoints used', '#View2 keypoints used',
                              '#landmarks in database'],
                     index=np.array(self.metadata)[:, 0]).to_pickle(filePath)

    def rectify_fusiello(self, d1=np.zeros(2), d2=np.zeros(2)):
        """
        Translation of Andre's IDL function rectify_fusello.
        """
        try:
            K1, R1, C1, _, _, _, _ = cv2.decomposeProjectionMatrix(self.view1.P)
            K2, R2, C2, _, _, _, _ = cv2.decomposeProjectionMatrix(self.view2.P)
        except:
            return

        C1 = cv2.convertPointsFromHomogeneous(C1.T).reshape(3, 1)
        C2 = cv2.convertPointsFromHomogeneous(C2.T).reshape(3, 1)

        oc1 = mdot(-R1.T, np.linalg.inv(K1), self.view1.P[:, 3])
        oc2 = mdot(-R2.T, np.linalg.inv(K2), self.view2.P[:, 3])

        v1 = (oc2-oc1).T
        v2 = np.cross(R1[2, :], v1)
        v3 = np.cross(v1, v2)

        R = np.array([v1/np.linalg.norm(v1), v2/np.linalg.norm(v2),
                      v3/np.linalg.norm(v3)]).reshape(3, 3)

        Kn1 = np.copy(K2)
        Kn1[0, 1] = 0
        Kn2 = np.copy(K2)
        Kn2[0, 1] = 0

        Kn1[0, 2] = Kn1[0, 2] + d1[0]
        Kn1[1, 2] = Kn1[1, 2] + d1[1]
        Kn2[0, 2] = Kn2[0, 2] + d2[0]
        Kn2[1, 2] = Kn2[1, 2] + d2[1]

        t1 = np.matmul(-R, C1)
        t2 = np.matmul(-R, C2)
        Rt1 = np.concatenate((R, t1), 1)
        Rt2 = np.concatenate((R, t2), 1)
        Prec1 = np.dot(Kn1, Rt1)
        Prec2 = np.dot(Kn2, Rt2)

        Tr1 = np.dot(Prec1[:3, :3], np.linalg.inv(self.view1.P[:3, :3]))
        Tr2 = np.dot(Prec2[:3, :3], np.linalg.inv(self.view2.P[:3, :3]))
        return Prec1, Prec2, Tr1, Tr2

    def estimate_pose_ls(self, X, frameDescriptors):
        """
        Estimate pose using Horn's method. Returns time taken to match
        descriptors with database, time taken to estimate pose, and number of
        3D landmarks used to estimate pose.
        """
        flag = 0
        # Match 3D points found in current frame with database
        matchDBStart = time.perf_counter()
        matches_db = self.match_descriptors(frameDescriptors,
                                            self.database.descriptors,
                                            matching_type='database')
        # frameIdx_raw and dbIdx_raw contain duplicate/unreliable matches. We
        # retain these indices as we add the landmarks with indices
        # complementary to these raw indices to the database. For pose
        # estimation however, we use frameIdx and dbIdx, which don't contain
        # duplicates.
        frameIdx_raw, dbIdx_raw = self.extract_match_indices(matches_db)
        self.matches_db_view1 = matches_db
        self.matches_db_view2 = matches_db

        matches_db = self.remove_duplicate_matches(matches_db)
        frameIdx, dbIdx = self.extract_match_indices(matches_db)

        matchDBTime = time.perf_counter() - matchDBStart

        poseEstStart = time.perf_counter()
        if (len(frameIdx) >= 3 and len(dbIdx) >= 3):
            XMatched = X[frameIdx]
            frameDescriptorsMatched = frameDescriptors[frameIdx]
            landmarksMatched = self.database.landmarks[dbIdx]
            landmarkDescriptorsMatched = self.database.descriptors[dbIdx]

            H = self.hornmm(XMatched, landmarksMatched)

            # Outlier removal, recalculate pose
            squErr = np.sqrt(np.sum(np.square(
                (XMatched.T - np.dot(H, landmarksMatched.T))), 0))

            # Absolute outlier rejection
            outliers = np.where(squErr > 3)[0]
            # Relative outlier rejection
            # outliers = detect_outliers(squErr)
            XMatched = np.delete(XMatched, outliers, axis=0)
            db_index_used = np.delete(dbIdx, outliers, axis=0)
            landmarksMatched = np.delete(landmarksMatched, outliers, axis=0)

            if (len(XMatched) >= 3 and len(landmarksMatched >= 3)):
                H = self.hornmm(XMatched, landmarksMatched)

                pose_change = np.abs(mat2vec(H) - self.currentPose)
                # Reject new pose if change from previous pose is too high
                if (pose_change > self.pose_threshold).any():
                    print('Pose change larger than threshold, returning' +
                          ' previous pose')
                    db_index_used = np.array([], dtype=int)
                    flag = 1
                else:
                    # self.database.trim(dbIdx[outliers])
                    self.currentPose = mat2vec(H)

                    # Add new landmarks to database
                    landmarksNew = np.delete(X, frameIdx_raw, axis=0)
                    landmarkDescriptorsNew = np.delete(frameDescriptors, frameIdx_raw,
                                                       axis=0)
                    # Transform new landmark positions to original pose
                    landmarksNew = mdot(np.linalg.inv(H), landmarksNew.T).T
                    self.database.update(landmarksNew, landmarkDescriptorsNew)
                    usedKeypoints = len(XMatched)
                    print('USED KEYPOINTS PARAM:', usedKeypoints)
                    print('DB INDEX USED:', len(db_index_used))
            else:
                print('Not enough matches with database, returning previous pose\n')
                db_index_used = np.array([], dtype=int)
                flag = 1
        else:
            print('Not enough matches with database, returning previous pose\n')
            db_index_used = np.array([], dtype=int)
            flag = 1
        poseEstTime = time.perf_counter() - poseEstStart

        return matchDBTime, poseEstTime, db_index_used, flag

    def estimate_pose_gn(self, X, frameDescriptors, in1, in2):
        """
        Matches keypoints from view1 and view2 to database indepedently. If
        there are matches, calls GN_estimation() to calculate pose. Retuns time
        taken to match keypoints with database, time taken to estimate pose,
        and number of keypoints in view1 and view2 used to calculte pose.
        """
        flag = 0
        matchDBStart = time.perf_counter()
        matches_view1db = self.match_descriptors(self.view1.descriptors,
                                                 self.database.descriptors,
                                                 matching_type='database')
        matches_view2db = self.match_descriptors(self.view2.descriptors,
                                                 self.database.descriptors,
                                                 matching_type='database')
        # frameIdx_raw and dbIdx_raw contain duplicate/unreliable matches. We
        # retain these indices as we add the landmarks with indices
        # complementary to these raw indices to the database. For pose
        # estimation however, we use frameIdx and dbIdx, which don't contain
        # duplicates.
        frameIdx1_raw, dbIdx1_raw = self.extract_match_indices(matches_view1db)
        frameIdx2_raw, dbIdx2_raw = self.extract_match_indices(matches_view2db)
        self.matches_db_view1 = matches_view1db
        self.matches_db_view2 = matches_view2db

        matches_view1db = self.remove_duplicate_matches(matches_view1db)
        matches_view2db = self.remove_duplicate_matches(matches_view2db)

        frameIdx1, dbIdx1 = self.extract_match_indices(matches_view1db)
        frameIdx2, dbIdx2 = self.extract_match_indices(matches_view2db)

        matchDBTime = time.perf_counter() - matchDBStart
        print('DB MATCHES VIEW1:', len(dbIdx1))
        print('DB MATCHES VIEW2:', len(dbIdx2))

        # Estimate pose
        if (len(frameIdx1) and len(frameIdx2)): # Landmarks seen in both views
            poseEstTime, key_index1, key_index2, \
            used_landmarks1, used_landmarks2, flag = \
                self.GN_estimation(
                    frameIdx1,
                    frameIdx2,
                    dbIdx1,
                    dbIdx2,
                )
        elif len(frameIdx1): # Landmarks seen in view 1 only
            poseEstTime, key_index1, key_index2, \
            used_landmarks1, used_landmarks2, flag = \
                self.GN_estimation(
                    frameIdx1,
                    np.array([], dtype=int),
                    dbIdx1,
                    np.array([], dtype=int),
                )
        elif len(frameIdx2): # Landmarks seen in view 2 only
            poseEstTime, key_index1, key_index2, \
            used_landmarks1, used_landmarks2, flag = \
                self.GN_estimation(
                    np.array([], dtype=int),
                    frameIdx2,
                    np.array([], dtype=int),
                    dbIdx2,
                )
        else:
            print('No matches with database, returning previous pose.\n')
            used_landmarks1 = np.array([], dtype=int)
            used_landmarks2 = np.array([], dtype=int)
            poseEstTime = 0
            flag = 1
            return matchDBTime, poseEstTime, used_landmarks1, used_landmarks2, \
                   flag

        H = vec2mat(self.currentPose)
        # Add new entries to database
        new_landmarks = []
        old_landmarks = []
        if len(in1) and flag == 0:
            for i in range(len(in1)):
                if (in1[i] not in frameIdx1_raw) and (in2[i] not in frameIdx2_raw):
                    new_landmarks.append(i)
                # elif (in1[i] in key_index1) and (in2[i] in key_index2):
                #     db_update_index1 = used_landmarks1[np.where(key_index1 == in1[i])[0]]
                #     db_update_index2 = used_landmarks2[np.where(key_index2 == in2[i])[0]]
                #     if db_update_index1 == db_update_index2:
                #         db_update_index = db_update_index1
                #         self.database.landmarks[db_update_index] = np.dot(np.linalg.inv(H), X[i, :].T).T
                #         self.database.descriptors[db_update_index] = frameDescriptors[i, :]

            X_new = X[new_landmarks, :]
            descriptors_new = frameDescriptors[new_landmarks, :]
            X_new = mdot(np.linalg.inv(H), X_new.T).T
            self.database.update(X_new, descriptors_new)

        return matchDBTime, poseEstTime, used_landmarks1, used_landmarks2, \
               flag

    def GN_estimation(self, key_index1, key_index2, db_index1,
                      db_index2, n_iterations=10, outlier_threshold=1):
        """
        Estimates pose using Gauss-Newton iterations. Based on Andre's IDL
        implentation numerical_estimation_2cams_v2.pro
        """
        P1 = self.view1.P
        P2 = self.view2.P
        key_coords1 = self.view1.key_coords[key_index1]
        key_coords2 = self.view2.key_coords[key_index2]
        # db_landmarks1 = self.database.landmarks[db_index1, :]
        # db_landmarks2 = self.database.landmarks[db_index2, :]

        usedKeypoints1 = len(db_index1)
        usedKeypoints2 = len(db_index2)

        poseEstStart = time.perf_counter()
        pose_est = self.currentPose

        for i in range(n_iterations):
            # print('Iteration number', i + 1)
            H = vec2mat(pose_est)
            J1, projections1, key_index1, db_index1 =  \
                self.iterate_jacobian(1, H, key_index1,
                                      db_index1,
                                      outlier_threshold, i)

            J2, projections2, key_index2, db_index2 = \
                self.iterate_jacobian(2, H, key_index2,
                                      db_index2,
                                      outlier_threshold, i)

            used_landmarks1 = len(db_index1)
            used_landmarks2 = len(db_index2)
            key_coords1 = self.view1.key_coords[key_index1]
            key_coords2 = self.view2.key_coords[key_index2]
            print('GN Iteration number:', i + 1)
            print('Used keypoints view1:', used_landmarks1)
            print('Used keypoints view2:', used_landmarks2,'\n')

            if used_landmarks1 + used_landmarks2 < 3:
                poseEstTime = time.perf_counter() - poseEstStart
                print('Cannot estimate pose from this frame, return last pose.')
                usedKeypoints1, usedKeypoints2 = 0, 0
                return poseEstTime, db_index1, db_index2, 1

            if J1.size and J2.size:
                J = np.concatenate((J1, J2), axis=0)
                e1 = (projections1[:2, :] - key_coords1.T).flatten(order='F')
                e2 = (projections2[:2, :] - key_coords2.T).flatten(order='F')
                e = np.concatenate((e1, e2))
            elif J1.size:
                J = J1
                e1 = (projections1[:2, :] - key_coords1.T).flatten(order='F')
                e = e1
            elif J2.size:
                J = J2
                e2 = (projections2[:2, :] - key_coords2.T).flatten(order='F')
                e = e2

            mean, variance = self.aggregate2var()
            if not (variance == 0).any():
                W = np.diag((1/np.sqrt(variance)))
            else:
                W = np.eye(6)

            A = np.dot(J.T, J) + np.dot(W.T, W)
            b = np.dot(J.T, e)

            pose_correction = np.linalg.lstsq(A, b, rcond=None)

            pose_est = pose_est - pose_correction[0]

        if self.filtering:
            F = self.state_transition # State transition model
            Q = self.process_covariance # Covariance of process noise
            H = self.observation_model # Observation model
            x_prior = np.dot(F, self.current_state)
            P_prior = mdot(F, self.state_covariance, F.T) + Q
            I = np.eye(12)
            R = np.linalg.inv(A) # Covariance of observation noise

            innovation = pose_est - np.dot(H, x_prior)
            S = R + mdot(H, P_prior, H.T)
            K = mdot(P_prior, H.T, np.linalg.inv(S))

            x_post = x_prior + np.dot(K, innovation)
            P_post = mdot((I - np.dot(K, H)), P_prior, (I - np.dot(K, H)).T) + mdot(K, R, K.T)

            pose_est = x_post[:6].copy()
            self.current_state = x_post
            self.pose_covariance = P_post

            # print('FILTERING!!!')
            # PLS = np.linalg.inv(np.dot(J.T, J))
            # prediction_est = np.dot(self.state_transtion, self.current_state)
            # prediction_cov = mdot(self.state_transition, )
            # covariance_est = self.pose_covariance + self.process_covariance
            # S = covariance_est + PLS
            # W = np.dot(covariance_est, np.linalg.inv(S))
            #
            # pose_est= self.currentPose + np.dot(W, (pose_est - self.currentPose))
            # self.pose_covariance = covariance_est - mdot(W, S, W.T)

        pose_change = np.abs(pose_est - self.currentPose)

        if (pose_change > self.pose_threshold).any():
            print('Pose change larger than threshold, returning' +
                  ' previous pose')
            db_index1 = np.array([])
            db_index2 = np.array([])
            flag = 1
        else:
            self.currentPose = pose_est
            flag = 0

        poseEstTime = time.perf_counter() - poseEstStart
        return poseEstTime, key_index1, key_index2, db_index1, db_index2, flag

    def iterate_jacobian(self, view_number, H, key_index, db_index,
                         outlier_threshold=2, iter_number=0):
        """
        Function called in main loop of GN_estimation.
        """
        if len(db_index):

            if view_number == 1:
                key_coords = self.view1.key_coords[key_index]
                P = self.view1.P
            elif view_number == 2:
                key_coords = self.view2.key_coords[key_index]
                P = self.view2.P

            db_landmarks = self.database.landmarks[db_index]
            projections = mdot(P, H, db_landmarks.T)
            projections = np.apply_along_axis(lambda v: v/v[-1], 0, projections)

            J = self.euler_jacobian(P, H, db_landmarks, projections)
            squErr = np.sqrt(np.sum(
                     np.square(projections[:2, :] - key_coords.T), 0))
            outliers = detect_outliers(squErr)

            if outliers.shape:
                key_index= np.delete(key_index, outliers, axis=0)
                projections = np.delete(projections, outliers, axis=1)
                db_index = np.delete(db_index, outliers, axis=0)
                Joutliers = np.array([2*outliers,
                                      (2*outliers + 1)]).flatten()
                J = np.delete(J, Joutliers, axis=0)
                squErr = np.delete(squErr, outliers)
            if iter_number >= 6:
                abs_outliers = np.where(squErr > outlier_threshold)[0]
                if len(abs_outliers) > 0:
                    key_index = np.delete(key_index, abs_outliers, axis=0)
                    projections = np.delete(projections, abs_outliers, axis=1)
                    db_index = np.delete(db_index, abs_outliers, axis=0)
                    Joutliers = np.array([2*abs_outliers,
                                          (2*abs_outliers + 1)]).flatten()
                    J = np.delete(J, Joutliers, axis=0)
        else:
            J = np.array([])
            projections = np.array([])
            key_index = np.array([], dtype=int)
        return J, projections, key_index, db_index

    @staticmethod
    def euler_jacobian(P, H, X, x):
        """
        Constructs the Euler Jacobian used in GN_estimation.
        """
        n_landmarks = X.shape[0]
        J = np.ones((2*n_landmarks, 6))

        for i in range(6):
            h = mat2vec(H)
            h[i] = h[i] + 0.5
            H_new = vec2mat(h)
            x_new = mdot(P, H_new, X.T)
            x_new = np.apply_along_axis(lambda v: v/v[-1], 0, x_new)

            J[:, i] = ((x_new - x)/0.5)[:2, :].flatten(order='F')
        return J

    def update_aggregate(self):
        """
        Welford's algorithm.
        Calculates online mean and M2, which is used to calculate the online
        variance. Use aggregate2var to convert aggregate to running mean and
        variance.
        """
        count, mean, M2 = self.aggregate
        count += 1
        delta = self.currentPose - mean
        mean = (mean + delta)/count
        delta2 = self.currentPose - mean
        M2 = M2 + delta * delta2
        self.aggregate = (count, mean, M2)

    def aggregate2var(self):
        """
        Returns mean and variance from aggregate variable.
        """
        count, mean, M2 = self.aggregate
        if count < 2:
            return mean, None
        else:
            variance = M2/(count - 1)
            return mean, variance

    @staticmethod
    def generate_dds(Tr1, Tr2):
        """
        Generates DD1 and DD2 for centering, used in conjunction with
        rectifyfusiello.
        """
        p = np.array([640, 480, 1], ndmin=2).T
        px = np.dot(Tr1, p)
        DD1 = (p[:2]-px[:2]) / px[2]
        px = np.dot(Tr2, p)
        DD2 = (p[:2]-px[:2]) / px[2]
        DD1[1] = DD2[1]
        return DD1, DD2

    def match_descriptors(self, descriptors1, descriptors2, matching_type=None):
        """
        Finds matches between two descriptor arrays, returns respective indices
        of matches.

        matching_type variable can be toggled to 'database' or 'intra-frame' to
        signal either database or intra-frame matching.
        """
        distRatio = self.distRatio
        try:
            if self.ratioTest:
                match = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
                matches = []
                for firstMatch, secondMatch in match:
                    if firstMatch.distance < distRatio*secondMatch.distance:
                        matches.append(firstMatch)
            else:
                matches = self.matcher.match(descriptors1, descriptors2)
        except ValueError:
            # If database only has one entry, knnMatch will throw a ValueError
            return []
        return matches

    @staticmethod
    def extract_match_indices(matches):
        """Returns query and train indices for a set of input matches."""
        # Returns empty arrays if matches is an empty list
        in1 = np.array([matches[i].queryIdx for i in range(len(matches))],
                       dtype='int')
        in2 = np.array([matches[i].trainIdx for i in range(len(matches))],
                       dtype='int')
        return in1, in2

    # Static methods used in process_frame()
    @staticmethod
    def triangulate_keypoints(P1, P2, coords1, coords2):
        """Outputs Nx4 array of homogeneous triangulated keypoint positions."""
        if len(coords1):
            X = cv2.triangulatePoints(P1, P2, coords1.T, coords2.T)
            return np.apply_along_axis(lambda v: v/v[-1], 0, X).T
        else:
            return np.array([])

    @staticmethod
    def remove_duplicate_matches(matches):
        matches = np.array(matches)
        matchIndices = np.array([(matches[i].queryIdx, matches[i].trainIdx)
                                 for i in range(len(matches))])
        if matchIndices.size:
            _, flags = np.unique(matchIndices[:, 1], return_counts=True)
        else:
            return matches
        return matches[np.where(flags == 1)[0]]

    @staticmethod
    def epipolar_constraint(coords1, coords2, T1, T2):
        n = coords1.shape[0]
        coords1H = np.hstack((coords1, np.ones((n, 1))))
        coords2H = np.hstack((coords2, np.ones((n, 1))))
        tcoords1 = np.dot(T1, coords1H.T)
        tcoords2 = np.dot(T2, coords2H.T)
        v1 = tcoords1[1, :]/tcoords1[2, :]
        v2 = tcoords2[1, :]/tcoords2[2, :]
        return np.where(abs(v1 - v2) <= 5)[0]

    @staticmethod
    def hornmm(X1, X2):
        """
        Translated from hornmm.pro

        Least squares solution to X1 = H*X2.
        Outputs H, the transformation from X2 to X1.
        Inputs X1 and X2 are Nx3 (or Nx4 homogeneous) arrays of 3D points.

        Implements method in "Closed-form solution of absolute orientation using
        unit quaternions",
        Horn B.K.P, J Opt Soc Am A 4(4):629-642, April 1987.
        """
        N = X2.shape[0]

        xc = np.sum(X2[:, 0])/N
        yc = np.sum(X2[:, 1])/N
        zc = np.sum(X2[:, 2])/N
        xfc = np.sum(X1[:, 0])/N
        yfc = np.sum(X1[:, 1])/N
        zfc = np.sum(X1[:, 2])/N

        xn = X2[:, 0] - xc
        yn = X2[:, 1] - yc
        zn = X2[:, 2] - zc
        xfn = X1[:, 0] - xfc
        yfn = X1[:, 1] - yfc
        zfn = X1[:, 2] - zfc

        sxx = np.dot(xn, xfn)
        sxy = np.dot(xn, yfn)
        sxz = np.dot(xn, zfn)
        syx = np.dot(yn, xfn)
        syy = np.dot(yn, yfn)
        syz = np.dot(yn, zfn)
        szx = np.dot(zn, xfn)
        szy = np.dot(zn, yfn)
        szz = np.dot(zn, zfn)

        M = np.array([[sxx, syy, sxz],
                      [syx, syy, syz],
                      [szx, szy, szz]])

        N = np.array([[(sxx+syy+szz), (syz-szy), (szx-sxz), (sxy-syx)],
                      [(syz-szy), (sxx-syy-szz), (sxy+syx), (szx+sxz)],
                      [(szx-sxz), (sxy+syx), (-sxx+syy-szz), (syz+szy)],
                      [(sxy-syx), (szx+sxz), (syz+szy), (-sxx-syy+szz)]])

        eVal, eVec = np.linalg.eig(N)
        index = np.argmax(eVal)
        vec = eVec[:, index]
        q0 = vec[0]
        qx = vec[1]
        qy = vec[2]
        qz = vec[3]

        X = np.array([[(q0*q0+qx*qx-qy*qy-qz*qz), 2*(qx*qy-q0*qz),
                       2*(qx*qz+q0*qy), 0],
                      [2*(qy*qx+q0*qz), (q0*q0-qx*qx+qy*qy-qz*qz),
                       2*(qy*qz-q0*qx), 0],
                      [2*(qz*qx-q0*qy), 2*(qz*qy+q0*qx),
                       (q0*q0-qx*qx-qy*qy+qz*qz), 0],
                      [0, 0, 0, 1]])

        Xpos = np.array([xc, yc, zc, 1])
        Xfpos = np.array([xfc, yfc, zfc, 1])
        d = Xpos - np.dot(np.linalg.inv(X), Xfpos)

        Tr = np.array([[1, 0, 0, -d[0]],
                       [0, 1, 0, -d[1]],
                       [0, 0, 1, -d[2]],
                       [0, 0, 0, 1]])
        return np.dot(X, Tr)


if __name__ == '__main__':
    view1 = CameraView()
    view2 = CameraView()
    sft = StereoFeatureTracker(view1, view2)
    import time

    def time_func(func):
        def timer(*args, **kwargs):
            start = time.perf_counter()
            rval = func(*args, **kwargs)
            end = time.perf_counter() - start
            print('That took {} seconds'.format(end))
            return rval
        return timer

    @time_func
    def init_bf():
        bf = cv2.BFMatcher()
