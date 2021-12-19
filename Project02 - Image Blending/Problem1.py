"""
Project 02

Name: ANTONY JOHN EDATHATTIL
Github: @antonyjohne

Dependencies: pip install PySimpleGUI
"""

import numpy as np 
import cv2
import PySimpleGUI as sg
import matplotlib.pyplot as plt 

def conv2(f, w):

	pad_r, pad_c = w.shape[0]//2, w.shape[1]//2

	#Recursive Function to Obtain Convolution of BGR Image of each channel seperately
	if len(f.shape)==3:
		b, g, r = cv2.split(f)
		b_conv = conv2(b, w)
		g_conv = conv2(g, w)
		r_conv = conv2(r, w)

		final_img = cv2.merge((b_conv,g_conv,r_conv))
		return final_img

	else:
		#Zero Padding Image
		rows, cols = f.shape
		img_pad = np.zeros([rows+(2*pad_r), cols+(2*pad_c)])#Initializing Padded Image
		conv_img = np.zeros([rows, cols])
		img_pad[pad_r:-pad_r, pad_c:-pad_c] = f

		#Convolution Process
		for i in range(rows):
			for j in range(cols):
				conv_img[i,j] = np.sum(w * img_pad[i:i+w.shape[0], j:j+w.shape[1]])
		
		return conv_img


##### Problem 1a
def ComputePyr(input_img, num_layers):

	#Initialize Empty Gaussian and Lapalcian Pyramid
	gPyr, lPyr = [], []
	temp_img = input_img
	gPyr.append(temp_img)

	#Compute Pyramids
	while (temp_img.shape[0]>1 and temp_img.shape[1]>1 and num_layers>0):

		#Smoothen and Subsample the Image by Scale 2
		gauss_img = conv2(temp_img, w)
		dscale_img = cv2.resize(gauss_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

		#Append Subsampled Image to Gaussian Pyramid
		gPyr.append(dscale_img)
		num_layers-=1
		temp_img = dscale_img

	for l in range(len(gPyr)-1, 0, -1):

		#Upsample the Image and Smoothen
		big_img = cv2.resize(gPyr[l], (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
		gauss_big_img = conv2(big_img, w)

		#Append the Laplacian of the Upsampled Image to Laplacian Pyramid
		lPyr.append(gPyr[l-1]-gauss_big_img)

	return (gPyr, lPyr)



def displayPyr(Pyr):

    for i in range(len(Pyr)):
        print(f"Pyramid Level: {i} -- Pyramid Shape: {Pyr[i].shape}")
        plt.imshow(Pyr[i], cmap='gray')
        plt.show()

#Function To Track Mouse Events in Image while Cropping
def crop_mouse_track(event, x, y, flags, param):

    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        #Draw Rectangle and Store Crop Boundaries
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


##### Problem 1b
#Browse and Load Image. Press 'c' to Confirm Crop, 'r' to Reset and Draw new Crop Rectangle
def GUI(grayscale = True):

	#Browse and Load Image
	file_types = [("JPEG (*.jpg)", "*.jpg"), ("PNG (*.png)", "*.png"), ("All files (*.*)", "*.*")]
	
	layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
    ]

	window = sg.Window("Open Image", layout)

	while True:
		event, values = window.read()
		if event == "Exit" or event == sg.WIN_CLOSED:
			break
		elif event == "Load Image":
			filename = values["-FILE-"]
			window.close()

	global image
	if grayscale==True:
		image = cv2.imread(filename, 0)

	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", crop_mouse_track)

	while True:
	    cv2.imshow("image", image)
	    key = cv2.waitKey(1) & 0xFF
	    #Reset Crop region if 'r' is pressed
	    if key == ord('r'):
	        image = clone.copy()
	    #Confirm Crop if 'c' is Pressed
	    elif key == ord('c'):
	        break

	#Return Mask and Cropped Section of the Image
	try:
		if len(refPt) == 2:

			cropped_img = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

			mask = np.zeros(image.shape[:2])
			cv2.rectangle(mask, refPt[0], refPt[1], 255, -1)

			cv2.imshow("Mask Image", mask)
			cv2.waitKey(0)
			return (cropped_img, mask)

	except NameError:
	    pass

	finally:
	    cv2.destroyAllWindows()


#Function to Collapse Blended Pyramid
def reconstruct(test_list):
	#Intialize Smallest Image
	final_img = ((gPyr_list_crop[0])*gPyr_list_source[-1]) + (((255-gPyr_list_crop[0]))*gPyr_list_target[-1])

	for i in range(len(test_list)):

		#Upscale and Smoothen Image. Add Smoothened Image to Next Level Laplacian 
		big_gauss = cv2.resize(final_img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
		gauss_i = conv2(big_gauss, w)
		final_img = (gauss_i+test_list[i])

	return final_img


#Initialize Source and Target Image (Change f number to merge different images)
#Choose Same Image as Source from GUI When running the Program
img_source = cv2.imread("f1.jpg",0)
img_target = cv2.imread("f2.jpg",0)

#Initialize the Gaussian Smoothening 2D Kernal
g_kernal = cv2.getGaussianKernel(5,1)
w = np.outer(g_kernal, g_kernal)

#Obtain Mask and Cropped Image From Source Image 
crop_source_img, mask = GUI()

#Compute the Laplacian and Gaussian Pyramids for Source, Target and Mask
layers_input = int(input("Enter the Number of Layers: ").strip())

print("\nComputing Pyramids of Source Image \n")
gPyr_list_source, lPyr_list_source = ComputePyr(img_source, layers_input)

print("Computing Pyramids of Target Image \n")
gPyr_list_target, lPyr_list_target = ComputePyr(img_target, layers_input)

print("Computing Pyramids of Mask Image \n")
gPyr_list_crop, lPyr_list_crop = ComputePyr(mask, layers_input)

blended_Pyr= []

#Reversing Crop list for Easier Blending Calculation
gPyr_list_crop.reverse()


##### Problem 1C
#Compute Blended Image using formula: LOutput = (G_mask)*LSource + (1-G_mask)*LTarget
print("Computing Pyramids of Blended Image \n")
for i in range(len(gPyr_list_crop)-1):
	blended_lPyr_img = ((gPyr_list_crop[i+1])*(lPyr_list_source[i])) + (((255-gPyr_list_crop[i+1]))*(lPyr_list_target[i]))			
	blended_Pyr.append(blended_lPyr_img)

#Collapse the Blended Laplacian Pyramid
output_final = reconstruct(blended_Pyr)


#Show all Output Images
plt.imshow(img_source, cmap='gray')
plt.title("Source Image")
plt.show()
plt.imshow(img_target, cmap='gray')
plt.title("Target Image")
plt.show()
plt.imshow(mask, cmap='gray')
plt.title("Mask Image")
plt.show()
plt.imshow(output_final, cmap='gray')
plt.title("Blended Image")
plt.show()	