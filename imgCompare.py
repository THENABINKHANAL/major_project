import cv2

img2 = cv2.imread('color_img02.jpg') 
img3 = cv2.imread('color_img03.jpg') 
img10 = cv2.imread('color_img010.jpg')

hist2=cv2.calcHist([img2], [0,1], None, [180,256], [0,180,0,256])
histogram_h2 = cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
hist3=cv2.calcHist([img3], [0,1], None, [180,256], [0,180,0,256])
histogram_h2 = cv2.normalize(hist3, hist3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
hist10=cv2.calcHist([img10], [0,1], None, [180,256], [0,180,0,256])
histogram_h2 = cv2.normalize(hist10, hist10, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

print(cv2.compareHist(hist2, hist3, cv2.HISTCMP_BHATTACHARYYA))
print(cv2.compareHist(hist2, hist10, cv2.HISTCMP_BHATTACHARYYA))

