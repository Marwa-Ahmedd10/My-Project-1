import cv2
import numpy as np
import glob
import os

# Calibration configuration
chessboard_size = (9, 6)  # For 9x6 inner corners
square_size = 1.0         # Change to actual square size if known (e.g., in cm)

# Termination criteria for sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ... for calibration
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store points
objpoints = []  # 3D points in world space
imgpoints = []  # 2D points in image plane

# Load images
image_paths = glob.glob("calibration_images/*.png")  # Dataset is PNG
for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(refined)

        # Optional: display detection
        cv2.drawChessboardCorners(img, chessboard_size, refined, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration result
np.savez("calibration_data.npz", 
         camera_matrix=camera_matrix, 
         dist_coeffs=dist_coeffs,
         rvecs=rvecs, 
         tvecs=tvecs)

print("Calibration successful!")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
