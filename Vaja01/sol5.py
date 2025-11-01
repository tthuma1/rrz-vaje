
### a)

# 0 0 0 0 0 0 0 1 0
# 0 0 0 1 0 0 0 1 1
# 0 1 0 1 0 0 1 1 1
# 1 1 1 1 0 1 0 0 0
# 1 0 0 1 1 0 0 0 0
# 0 0 0 0 0 0 0 0 1
# 0 0 0 0 0 0 1 1 1
# 0 1 0 0 0 1 1 1 1
# 1 1 0 0 0 0 0 0 1

# Prvi prehod:

# 0  0 0 0 0 0  0 1 0
# 0  0 0 2 0 0  0 1 1
# 0  3 0 2 0 0  4 1 1
# 5  3 3 2 0 6  0 0 0
# 5  0 0 2 2 0  0 0 0
# 0  0 0 0 0 0  0 0 7
# 0  0 0 0 0 0  8 8 7
# 0  9 0 0 0 10 8 8 7
# 11 9 0 0 0 0  0 0 7

# konflikti = ((4,1), (5,3), (3,2), (7,8), (10,8), (11, 9))
# urejeni konflikti = ((4,1), (5,3,2) (7,8,10) (11, 9))

# Drugi prehod:

# 0 0 0 0 0 0 0 1 0
# 0 0 0 2 0 0 0 1 1
# 0 2 0 2 0 0 1 1 1
# 2 2 2 2 0 6 0 0 0
# 2 0 0 2 2 0 0 0 0
# 0 0 0 0 0 0 0 0 7
# 0 0 0 0 0 0 7 7 7
# 0 9 0 0 0 7 7 7 7
# 9 9 0 0 0 0 0 0 7



# I = I.astype(np.uint8)
# k = np.array()
# k = k.astype(np.uint8)
             
# er = cv2.erode(I, k)
# er = cv2.dilate(I, k)

# cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) namesto ročno np.array()
# cv2.open() in cv2.close() za opening in closing