#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')




#left_image = cv.imread('l6.png', cv.IMREAD_GRAYSCALE)
#right_image = cv.imread('r6.png', cv.IMREAD_GRAYSCALE)

left_image = cv.imread('Left.jpg', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('right.jpg', cv.IMREAD_GRAYSCALE)

#left_image = cv.imread('KLeft.png', cv.IMREAD_GRAYSCALE)
#right_image = cv.imread('Kright.png', cv.IMREAD_GRAYSCALE)

min_disp=0
num_disp=96



stereo = cv.StereoBM_create(numDisparities=96, blockSize=21)
# For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to numDisparities.
# The search range can then be shifted by changing the minimum disparity.
# larger block size implies smoother, though less accurate disparity
disparity_map = stereo.compute(left_image, right_image)

cv.imshow("left", left_image)
cv.imshow("right", right_image)

plt.imshow(disparity_map)
plt.axis('off')
plt.show()


'''
h, w = left_image.shape[:2]
focal_length = 0.8 * w


# Perspective transformation matrix
Q = np.float32([[1, 0, 0, -w / 2.0],
                [0, -1, 0, h / 2.0],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])


points_3D = cv.reprojectImageTo3D(disparity_map, Q)
colors = cv.cvtColor(left_image, cv.COLOR_BGR2RGB)
mask_map = disparity_map > disparity_map.min()
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

create_output(output_points, output_colors, 'depthmap.ply')

cv.imshow('Left Image', left_image)
cv.imshow('Right Image', right_image)
cv.imshow('Disparity Map', (disparity_map - min_disp) / num_disp)
cv.waitKey()
cv.destroyAllWindows()

'''

