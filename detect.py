from calibrate import load_coefficients
import numpy as np
import time
import cv2

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters_create()

matrix_coeffs, dist_coeff = load_coefficients()

source = cv2.imread('calibration/calibration1.jpg')

prev_frame_time, new_frame_time = 0, 0

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    out = frame.copy()
    im_w, im_h = out.shape[:2]

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=params, 
        cameraMatrix=matrix_coeffs, distCoeff=dist_coeff)
    new_frame_time = time.time()
    if len(corners) == 4:
        ids = ids.flatten()
        ref_pts = []
        
        # Loop over ids in topleft, topright, bottomright, bottomleft order    
        for i in (97, 208, 2, 24):
            # Grab idx of corner w current id and append corner (x, y) coords
            # to our list of reference points
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(corners[j])
            ref_pts.append(corner)

        # Unpack ArUco reference points, use them to define destination transform matrix
        # Points in topleft, topright, bottomright, bottomleft order
        (ref_pt_tl, ref_pt_tr, ref_pt_br, ref_pt_bl) = ref_pts
        dst_mat = [ref_pt_tl[1], ref_pt_tr[2], ref_pt_br[2], ref_pt_bl[0]]
        dst_mat = np.array(dst_mat)

        # Get dimensions of source image and define transformation matrix for it in same order
        # Source image is twitch stream in our case
        (src_h, src_w) = source.shape[:2]
        src_mat = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

        # Compute homography matrix and warp the source image to the dest based on the homography
        (h, _) = cv2.findHomography(src_mat, dst_mat)
        warped = cv2.warpPerspective(source, h, (im_h, im_w))

        # Construct a mask for source image now that the perspective warp has taken place
        # Need this to copy source image into destination
        mask = np.zeros((im_w, im_h), dtype='uint8')
        cv2.fillConvexPoly(mask, dst_mat.astype('int32'), (255, 255, 255), cv2.LINE_AA)

        # Gives source image a black border around it
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, rect, iterations=2)

        # Create 3 channel version of mask by stacking it depth wise, s.t we can copy
        # the warped source image into the input image
        mask_scaled = mask.copy() / 255.0
        mask_scaled = np.dstack([mask_scaled] * 3)

        # Copy the warped src image into the input frame by:
        # Multiplying the warped image and masked together
        # Multiplying the original input image with the mask (giving more weight to the input
        # where there are NOT masked pixels) and
        # adding the resulting multiplications together
        warped_multiplied = cv2.multiply(warped.astype('float'), mask_scaled)
        img_multiplied = cv2.multiply(out.astype('float'), 1.0 - mask_scaled)
        out = cv2.add(warped_multiplied, img_multiplied)
        out = out.astype('uint8')

    else:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)             
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = str(int(fps))

    cv2.putText(out, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('out', out)
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break