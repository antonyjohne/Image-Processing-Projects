import numpy as np
import cv2
import matplotlib.pyplot as plt

#Problem 2a
#Histogram Equalization Function
def hist_eq(img_gray):
	img_flatten = img_gray.flatten()
	min_px, max_px = min(img_flatten), max(img_flatten)

	new_img = ((img_gray - min_px)/(max_px-min_px))*255
	new_img = np.array(new_img, dtype=np.uint8)

	#Drawing Histograms (Comparison)
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	plt.subplot(1,2,1)
	plt.title("Original Hist.")
	plt.xlabel("Intensities")
	plt.ylabel("No. of Pixels")
	plt.hist(img_gray.flatten(), 256, [0,255], label="Original Hist.")

	plt.subplot(1,2,2)
	plt.title("Equalized Hist.")
	plt.hist(new_img.flatten(), 256, [0,255], color="Orange")
	plt.xlabel("Intensities")
	plt.ylabel("No. of Pixels")
	plt.show()

	return new_img

#2D DFT Function using np.fft
def DFT2(f):

	#Initializing Empty Complex Matrix
	row, col = f.shape 
	f_2DFFT = np.zeros([row, col], dtype=complex)

	#Calculating 1-D FFT of Rows
	for i in range(row):
			f_2DFFT[i] = np.fft.fft(f[i])

	#Calculating 1-D FFT of Columns 
	f_2DFFT = f_2DFFT.T
	for j in range(col):
			f_2DFFT[j] = np.fft.fft(f_2DFFT[j])

	#Obtaining Original Matrix and Shifting Center
	f_2DFFT = f_2DFFT.T
	f_2DFFT = np.fft.fftshift(f_2DFFT)

	return f_2DFFT


#Problem 2b
#2D inverse FFT Function using DFT2 Function
def iDFT2(f):

	row, col = f.shape 
	conj_f = f.conjugate()
	f_iFFT = DFT2(conj_f)
	f_iFFT = f_iFFT.conjugate()/(row*col)

	iFFT = np.fft.fftshift(f_iFFT)

	return iFFT

#Read Image and Convert to Grayscale
img = cv2.imread("lena.png")
f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

f_row, f_col = f.shape

#Obtain Equalized Image between 0-1
hist_eq_f = hist_eq(f)
cv2.imshow("Equalized Image", hist_eq_f)
cv2.waitKey(0)


#FFT2 of Image
F = DFT2(hist_eq_f)

# # Testing Code for FFT2 (Uncomment to Verify)
# fft2_np = np.fft.fftshift(np.fft.fft2(hist_eq_f))
# for i in range(f_row):
# 	for j in range(f_col):
# 		if(F[i,j] != fft2_np[i,j]):
# 			print("False")


#Inverst FFT2 of Image
G = iDFT2(F)

# # Testing Code for iFFT2 (Uncomment to Verify)
# ifft2_np = np.fft.ifft2(F)
# for i in range(f_row):
# 	for j in range(f_col):
# 		if(G[i,j] != ifft2_np[i,j]):
# 			print("False")


#Plot Magnitude and Phase of FFT Image
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1,2,1)	
plt.imshow(np.log(1+np.abs(F)), cmap='gray')
plt.title("2D FFT (Centered Magnitude Spectrum)")

plt.subplot(1,2,2)
plt.imshow(np.log(1+np.abs(np.angle(F))), cmap='gray')
plt.title("2D FFT (Centered Phase)")
plt.show()

#Plot Magnitude and Phase of Inverse FFT of Image
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1,2,1)	
plt.imshow(np.abs(G.real), cmap='gray')
plt.title("2D iFFT (Centered Magnitude Spectrum)")

plt.subplot(1,2,2)
plt.imshow(np.angle(G), cmap='gray')
plt.title("2D iFFT (Centered Phase)")
plt.show()


#Verify f-G == 0
G_real = hist_eq_f - abs(G.real)
G_real = np.array(G_real, dtype=np.uint8)
print("f-G Result Matrix \n\n", G_real)

cv2.imshow("f-G Result", G_real)
cv2.waitKey(0)
cv2.destroyAllWindows()