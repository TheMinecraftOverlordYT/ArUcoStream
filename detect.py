from calibrate import load_coefficients
import numpy as np
import time
import cv2

marker_ids = [97, 208, 2, 24]


def find_markers(img, c, d, marker_size=6, total_markers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    ar_dict = cv2.aruco.Dictionary_get(key)
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, ar_dict, parameters=params,
                                                     cameraMatrix=c, distCoeff=d)

    if draw:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

    return corners, ids, rejected


def augment(corners, ids, img, img_aug):
    ids = ids.flatten()
    ref_pts = []
    global marker_ids
    # Loop over ids in topleft, topright, bottomright, bottomleft order
    for i in marker_ids:
        # Grab idx of corner w current id and append corner (x, y) coords
        # to our list of reference points
        j = np.squeeze(np.where(ids == i))
        corner = np.squeeze(corners[j])
        ref_pts.append(corner)

    # Unpack ArUco reference points, use them to define destination transform matrix
    # Points in topleft, topright, bottomright, bottomleft order
    ref_pt_tl, ref_pt_tr, ref_pt_br, ref_pt_bl = ref_pts
    dst_mat = [ref_pt_tl[1], ref_pt_tr[2], ref_pt_br[2], ref_pt_bl[0]]
    dst_mat = np.array(dst_mat)

    # Get dimensions of source image and define transformation matrix for it in same order
    # Source image is twitch stream in our case
    src_h, src_w = img_aug.shape[:2]
    src_mat = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

    # Compute homography matrix and warp the source image to the dest based on the homography
    h, _ = cv2.findHomography(src_mat, dst_mat)

    img_out = cv2.warpPerspective(img_aug, h, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, dst_mat.astype(int), (0, 0, 0))

    img_out = img + img_out
    return img_out


def main():
    cap = cv2.VideoCapture(0)
    camera_matrix, dist_matrix = load_coefficients()
    img_aug = cv2.imread('calibration/calibration1_undistorted.jpg')

    new_frame_time, prev_frame_time = 0, 0

    while True:
        new_frame_time = time.time()
        success, frame = cap.read()
        corners, ids, rejected = find_markers(frame, camera_matrix, dist_matrix)

        if len(corners) == 4:
            frame = augment(corners, ids, frame, img_aug)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Image', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
