import cv2
import numpy as np
import random as rand
import math
import os
import matplotlib.pyplot as plt

# Function to plot histogram of image	
def plot_hist_graph(decrypted_array):

	# Numpy array needs to be typecast into float32 to adjust depth before conversion to grayscale
	decrypted_array = np.float32(decrypted_array)
	decrypted_array = cv2.cvtColor(decrypted_array, cv2.COLOR_BGR2GRAY)
	
	plt.hist(decrypted_array.flatten(), bins=256, color='skyblue')
	plt.title("Histogram for {} with {} map decryption".format(choice_image.title(), choice_map))
	plt.xlabel("Pixel intensity")
	plt.ylabel("Pixel frequency")
	
	try:
		os.mkdir("./decrypted_images/graphs/")
		os.mkdir("./decrypted_images/graphs/histograms")	
		plt.savefig("./decrypted_images/graphs/histograms/{}_histogram_{}_map.png".format(choice_image, choice_map))
	
	except:
		plt.savefig("./decrypted_images/graphs/histograms/{}_histogram_{}_map.png".format(choice_image, choice_map))
		
	plt.show()
	
# Function to plot correlation graph of image	
def plot_corr_graphs(decrypted_img, original_img):
	
	# Convert images to int type
	decrypted_img =  np.float32(decrypted_img)
	
	# Read images as grayscale
	decrypted_img = cv2.cvtColor(decrypted_img, cv2.COLOR_BGR2GRAY)
	original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	
	spectra = ['vertical', 'horizontal', 'diagonal']
	height, width = decrypted_img.shape
	
	# Initialize empty numpy arrays to random pixel values
	a = np.zeros(4096)
	x = np.zeros(4096)
	y = np.zeros(4096)
	z = np.zeros(4096)
	
	for i in range(4096):
		random_y = rand.randint(0, height-2)
		random_x = rand.randint(0, width-2)
		
		a[i] = decrypted_img[random_y][random_x]
		
		if random_y+1 != height or random_x+1 != width:		# to ensure array does not exceed upper limit
			x[i] = original_img[random_y][random_x+1]		# for horizontal coefficient
			y[i] = original_img[random_y+1][random_x]		# for vertical coefficient
			z[i] = original_img[random_y+1][random_x+1]	# for diagonal coefficient
		
		elif random_y != 0 or random_x != 0:			# to ensure array does not exceed lower limit
			x[i] = original_img[random_y][random_x-1]	
			y[i] = original_img[random_y-1][random_x]	
			z[i] = original_img[random_y-1][random_x-1]	
		else:
			i = i-1		# To make it simple by starting over
	
	for spectrum in spectra:	# for all 3 spectra
		if spectrum == 'horizontal':
			temp_array = x
		elif spectrum == 'vertical':
			temp_array = y
		else:
			temp_array = z
		
		# Calculate correlation coefficient
		corr_matrix = np.corrcoef(temp_array, a)
		print(corr_matrix)
		
		# Plot graph
		plt.scatter(temp_array, a, s=5)
		plt.title("Correlation Plot in {} spectrum for {} with {} map decryption\nCorrelation coefficient = {}".format(spectrum.title(), choice_image.title(), choice_map.title(), corr_matrix[0][1]))
		plt.xlabel("Original Image")
		plt.ylabel("Image with decryption")
		
		try:
			os.mkdir("./decrypted_images/graphs/corr_graphs")
			plt.savefig("./decrypted_images/graphs/corr_graphs/{}_{}_correlation_{}_map.png".format(spectrum, choice_image, choice_map))
		
		except:
			plt.savefig("./decrypted_images/graphs/corr_graphs/{}_correlation_{}_map.png".format(spectrum, choice_image, choice_map))
			
		plt.show()
	
# Select image and chaotic map
while True: 
	print("Select one image: ")
	print("1. Lena")
	print("2. Baboon")
	print("3. Peppers")
	choice_image = input("Your choice: ")
	
	if choice_image == '1' or choice_image.casefold() in 'lena':
		choice_image = 'lena'
		
	elif choice_image == '2' or choice_image.casefold() in 'baboon':
		choice_image = 'baboon'
		
	elif choice_image == '3' or choice_image.casefold() in 'peppers':
		choice_image = 'peppers'
		
	else:
		print("Invalid choice, try again.\n")
		break
	
	print("Select chaotic map: ")
	print("1. Logistic")
	print("2. Tent")
	print("3. Sine")
	choice_map = input("Your choice: ")
	
	if choice_map == '1' or choice_map.casefold() in 'logistic':
		choice_map = 'logistic'
		r = 3.9
		break
		
	elif choice_map == '2' or choice_map.casefold() in 'tent':
		choice_map = 'tent'
		r = 1.8
		break
		
	elif choice_map == '3' or choice_map.casefold() in 'sine':
		choice_map = 'sine'
		r = 3.9
		break

# Read image and its shape
chaotic_img_array = cv2.imread("./encrypted_images/{}_encrypted_{}.png".format(choice_map, choice_image))
original_image = cv2.imread("../{}.jpg".format(choice_image))

# Convert PNG image into numpy array 
height, width, channels = chaotic_img_array.shape

n = height*width*channels

# Convert image into flat array
flat_array = chaotic_img_array.flatten()

# Initialize array for chaotic sequence
x = np.zeros(n)
x[0] = 0.1

# Initialize array for decrypted image
decrypted_array = np.zeros(n)

# Encrypt image pixel-wise using XOR operation
for i in range(1, n):
	
	decrypted_array[i-1] = flat_array[i-1] ^ int(x[i-1]*255)

	if choice_map == 'logistic':
		x[i] = r * x[i-1] * (1 - x[i-1])
		
	elif choice_map == 'tent':
		if x[i-1] < 0.5:
			x[i] = r * x[i-1]
		else:
			x[i] = r * (1-x[i-1])
		
	elif choice_map == 'sine':
		x[i] = (r * abs(math.sin(x[i-1]))) % 1 

# Reshape decrypted array according to original shape
decrypted_image = np.reshape(decrypted_array, (height, width, channels))

# Save file
try:
	os.mkdir("./decrypted_images/")
	cv2.imwrite("./decrypted_images/{}_decrypted_{}.jpg".format(choice_map, choice_image), decrypted_image)
except:
	cv2.imwrite("./decrypted_images/{}_decrypted_{}.jpg".format(choice_map, choice_image), decrypted_image)

# Ask if they want graphs to be plotted
choice_graphs = input("Do you want to plot graphs for this image? (Y/n) ")

if choice_graphs.casefold() == 'y':
	plot_hist_graph(decrypted_image)
	plot_corr_graphs(decrypted_image, original_image)
