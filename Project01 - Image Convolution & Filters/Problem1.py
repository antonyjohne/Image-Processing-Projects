"""
UnityID = aedatha
Name: ANTONY JOHN EDATHATTIL
Student ID: 200375601

Uncomment mentioned lines in the program to obtain outputs with the desired filter.
"""


import numpy as np 
import cv2
import matplotlib.pyplot as plt

#Problem 1a
def conv2(f, w, pad):

	#Recursive Function to Obtain Convolution of BGR Image of each channel seperately
	if len(f.shape)==3:
		b, g, r = cv2.split(f)
		b_conv = conv2(b, w, pad)
		g_conv = conv2(g, w, pad)
		r_conv = conv2(r, w, pad)

		final_img = cv2.merge((b_conv,g_conv,r_conv))
		return final_img

	else:
		row, col = f.shape
		img_pad = np.zeros([row+2,col+2]) #Initializing Padded Image
		ip_row, ip_col = img_pad.shape
		img_pad[1:-1, 1:-1] = f 		  #Set Padded Image Center to Image
		img_copy = np.zeros([row,col])

		#Zero Padding
		if pad == 0:
			pass
			title = "Zero Padded Image"

		#Wrap Around Padding
		elif pad == 1:
			#Set Rows
			img_pad[1:-1, 0] = f[:, col-1]
			img_pad[1:-1, ip_col-1] = f[:, 0]

			#Set Columns
			img_pad[0, 1:-1] = f[row-1, :]
			img_pad[ip_row-1, 1:-1] = f[0, :]

			#Set Corner Values
			img_pad[0,0], img_pad[ip_row-1, ip_col-1] = f[row-1, col-1], f[0,0]
			img_pad[0,ip_col-1], img_pad[ip_row-1, 0] = f[row-1, 0], f[0,col-1]

			title = "Wrap Around Padded Image"

		#Copy-Edge Padding
		elif pad == 2:
			#Set Rows
			img_pad[1:-1, 0] = f[:, 0]
			img_pad[1:-1, ip_col-1] = f[:, col-1]

			#Set Columns
			img_pad[0, 1:-1] = f[0, :]
			img_pad[ip_row-1, 1:-1] = f[row-1, :]

			#Set Corner Values
			img_pad[0,0], img_pad[ip_row-1, ip_col-1] = f[0,0], f[row-1, col-1]
			img_pad[0,ip_col-1], img_pad[ip_row-1, 0] = f[0,col-1], f[row-1, 0] 

			title = "Copy-Edge Padded Image"

		#Reflection Padding
		elif pad == 3:
			#Set Rows
			img_pad[1:-1, 0] = f[:, 1]
			img_pad[1:-1, ip_col-1] = f[:, col-2]

			#Set Columns
			img_pad[0, 1:-1] = f[1, :]
			img_pad[ip_row-1, 1:-1] = f[row-2, :]

			#Set Corner Values
			img_pad[0,0], img_pad[ip_row-1, ip_col-1] = f[1,1], f[row-2, col-2]
			img_pad[0,ip_col-1], img_pad[ip_row-1, 0] = f[1,col-2], f[row-2, 1] 

			title = "Reflection Padded Image"

		#Convolution Process
		for i in range(row):
			for j in range(col):
				img_copy[i,j] = np.sum(w * img_pad[i:i+w.shape[0], j:j+w.shape[1]])
		return img_copy


#Read Image (Uncomment for Testing Wolf Image)
img = cv2.imread("lena.png")
# img = cv2.imread("wolves.png")
print("\nInitializing Filters and Performing Spatial Convolution. Please Wait...\n")

#Image BGR to Gray (Uncomment for Testing Gray Image)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Initializing Different Filters
box = (1/9) * np.array([[1,1,1],[1,1,1],[1,1,1]])
dMx = np.array([[-1,1]])
dMya = np.array([[-1],[1]])
dMyb = np.array([[1],[-1]])
prewittMx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittMy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
sobelMx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobelMy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
robertsMx = np.array([[0,1],[-1,0]])
robertsMy = np.array([[1,0],[0,-1]])

#Performing Convolution for Various Filters
oimg_box = conv2(img, box, 0)
oimg_dMx = conv2(img, dMx, 1)
oimg_dMya = conv2(img, dMya, 1)
oimg_dMyb = conv2(img, dMyb, 0)
oimg_prewittMx = conv2(img, prewittMx, 2)
oimg_prewittMy = conv2(img, prewittMy, 2)
oimg_sobelMx = conv2(img, sobelMx, 3)
oimg_sobelMy = conv2(img, sobelMy, 3)
oimg_robertsMx = conv2(img, robertsMx, 0)
oimg_robertsMy= conv2(img, robertsMy, 0)

#Filter Edge Detection
D_final = np.sqrt(np.square(oimg_dMx) + np.square(oimg_dMya))
prewitt_final = np.sqrt(np.square(oimg_prewittMx) + np.square(oimg_prewittMy))
sobel_final = np.sqrt(np.square(oimg_sobelMx) + np.square(oimg_sobelMy))
roberts_final = np.sqrt(np.square(oimg_robertsMx) + np.square(oimg_robertsMy))


############ Uncomment below code while Using "Lena.png" Image #####################
############ Comment out below code while Using "Wolves.png" Image #################

# Stacking Images for Comparison
box_stack = np.hstack([img, oimg_box])
D_stack = np.hstack([oimg_dMx, oimg_dMya, D_final])
prewitt_stack = np.hstack([oimg_prewittMx, oimg_prewittMy, prewitt_final])
sobel_stack = np.hstack([oimg_sobelMx, oimg_sobelMy, sobel_final])
roberts_stack = np.hstack([oimg_robertsMx, oimg_robertsMy, roberts_final])


#Display All Filters
cv2.imshow("Box Filter", np.array(box_stack, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("1D Filter", np.array(D_stack, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Prewitt Filter", np.array(prewitt_stack, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Sobel Filter", np.array(sobel_stack, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Roberts Filter", np.array(roberts_stack, dtype=np.uint8))
cv2.waitKey(0)

######################################################################################


############ Uncomment below code while Using "Wolves.png" Image #####################
############ Comment out below code while Using "Lena.png" Image #################

"""
cv2.imshow("Box Filter", np.array(oimg_box, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("1D Filter Mx", np.array(oimg_dMx, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("1D Filter Mya", np.array(oimg_dMya, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Sobel Filter Mx", np.array(oimg_sobelMx, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Sobel Filter My", np.array(oimg_sobelMy, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Prewitt Filter Mx", np.array(oimg_prewittMx, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Prewitt Filter My", np.array(oimg_prewittMy, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Roberts Filter Mx", np.array(oimg_robertsMx, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Roberts Filter My", np.array(oimg_robertsMy, dtype=np.uint8))
cv2.waitKey(0)

cv2.imshow("1D Filter", np.array(D_final, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Prewitt Filter", np.array(prewitt_final, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Sobel Filter", np.array(sobel_final, dtype=np.uint8))
cv2.waitKey(0)
cv2.imshow("Roberts Filter", np.array(roberts_final, dtype=np.uint8))
cv2.waitKey(0)

"""
######################################################################################


#Problem 1b
#Initializing Image
initial_img = np.zeros([1024,1024])
initial_img[512,512] = 255

#Performing Convolution
print("Performing Box Filter Convolution on 1024x1024 Unit Impulse Image \n\n")
output_img = conv2(initial_img, box, 0)
output_img = np.array(output_img, dtype=np.uint8)
print("Center 5x5 Matrix", "\n\n", output_img[510:515,510:515])

#Initial Image and Output Image Display
cv2.imshow("Initial Unit Impulse", initial_img)
cv2.waitKey(0)

cv2.imshow("Convoluted Box Filter Unit Impulse", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()