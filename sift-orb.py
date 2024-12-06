import cv2
import numpy as np

def load_images(genuine_logo_path, suspect_logo_path):
    genuine_logo = cv2.imread(genuine_logo_path, cv2.IMREAD_GRAYSCALE)
    suspect_logo = cv2.imread(suspect_logo_path, cv2.IMREAD_GRAYSCALE)
    return genuine_logo, suspect_logo

def detect_and_compute_sift(image):
    sift = cv2.SIFT_create(contrastThreshold=0.02)  # Lowering the contrast threshold to detect more keypoints
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def detect_and_compute_orb(image):
    orb = cv2.ORB_create(nfeatures=2000)  # Increasing the number of features for ORB
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def compute_homography(keypoints1, keypoints2, matches):
    if len(matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)  # Reduced threshold for more precision
        return H, mask
    return None, None

def focus_on_roi(image, roi):
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]
    return roi_image

def preprocess_image(image):
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compute_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def resize_images_to_match(img1, img2):
    height, width = img1.shape[:2]
    img2_resized = cv2.resize(img2, (width, height))
    return img1, img2_resized

def overlay_images(genuine_logo, suspect_logo, alpha=0.5):
    genuine_logo_resized, suspect_logo_resized = resize_images_to_match(genuine_logo, suspect_logo)
    overlay = cv2.addWeighted(genuine_logo_resized, alpha, suspect_logo_resized, 1 - alpha, 0)
    return overlay

def compare_logos_comprehensive(genuine_logo_path, suspect_logo_path, roi=None):
    genuine_logo, suspect_logo = load_images(genuine_logo_path, suspect_logo_path)

    if roi:
        genuine_logo_roi = focus_on_roi(genuine_logo, roi)
        suspect_logo_roi = focus_on_roi(suspect_logo, roi)
    else:
        genuine_logo_roi = genuine_logo
        suspect_logo_roi = suspect_logo

    keypoints1, descriptors1 = detect_and_compute_sift(genuine_logo_roi)
    keypoints2, descriptors2 = detect_and_compute_sift(suspect_logo_roi)

    good_matches = match_keypoints(descriptors1, descriptors2)
    print(f"Number of good matches: {len(good_matches)}")

    H, mask = compute_homography(keypoints1, keypoints2, good_matches)

    matched_image = cv2.drawMatches(genuine_logo_roi, keypoints1, suspect_logo_roi, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Matched Keypoints', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    contours1 = preprocess_image(genuine_logo_roi)
    contours2 = preprocess_image(suspect_logo_roi)

    if len(contours1) > 0 and len(contours2) > 0:
        similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0.0)
        hu_moments1 = compute_hu_moments(contours1[0])
        hu_moments2 = compute_hu_moments(contours2[0])

        diff = np.sum(np.abs(hu_moments1 - hu_moments2))

        print(f"Shape similarity score: {similarity}")
        print(f"Hu Moments difference: {diff}")

        if similarity >= 0.1 or diff > 0.1:
            print(f"The suspect logo is likely a fake. (Good Matches: {len(good_matches)})")
        else:
            print(f"The suspect logo is likely genuine. (Good Matches: {len(good_matches)})")

    overlay = overlay_images(genuine_logo_roi, suspect_logo_roi)
    cv2.imshow('Overlay Comparison', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    roi = (50, 50, 100, 100)  # Adjust based on the area of interest
    compare_logos_comprehensive('original2.png', 'fake2.png', roi)
