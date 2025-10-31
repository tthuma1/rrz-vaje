
I = I.astype(np.uint8)
k = np.array()
k = k.astype(np.uint8)
             
er = cv2.erode(I, k)
er = cv2.dilate(I, k)

# cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) namesto ročno np.array()
# cv2.open() in cv2.close() za opening in closing