import cv2
import numpy as np
import random as rand
import math
import os
import matplotlib.pyplot as plt

# Function to plot histogram of image	
def plot_hist_graph(encrypted_array):
	
	plt.hist(encrypted_array, bins=256, color='skyblue')
	plt.title("Histogram for {} with {} map encryption".format(choice_image.title(), choice_map))
	plt.xlabel("Pixel intensity")
	plt.ylabel("Pixel frequency")
	
	try:
		plt.savefig("./encrypted_images/graphs/histograms/{}_histogram_{}_map.png".format(choice_image, choice_map))
	
	except:
		os.mkdir("./encrypted_images/graphs/histograms")	#TO DO: Make this command work
		plt.savefig("./encrypted_images/graphs/histograms/{}_histogram_{}_map.png".format(choice_image, choice_map))
		
	plt.show()
	
# Function to plot correlation graph of image	
def plot_corr_graphs(encrypted_img, original_img):
	
	# Convert images to int type
	encrypted_img =  np.float32(encrypted_img)
	
	# Read images as grayscale
	encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2GRAY)
	original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	
	spectra = ['vertical', 'horizontal', 'diagonal']
	height, width = encrypted_img.shape
	
	# Initialize empty numpy arrays to random pixel values
	a = np.zeros(4096)
	x = np.zeros(4096)
	y = np.zeros(4096)
	z = np.zeros(4096)
	
	for i in range(4096):
		random_y = rand.randint(0, height-2)
		random_x = rand.randint(0, width-2)
		
		a[i] = encrypted_img[random_y][random_x]
		
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
		plt.title("Correlation Plot in {} spectrum\nfor {} with {} map encryption\nCorrelation coefficient = {}".format(spectrum.title(), choice_image.title(), choice_map.title(), corr_matrix[0][1]))
		plt.xlabel("Image without encryption")
		plt.ylabel("Image with encryption")
		
		try:
			plt.savefig("./encrypted_images/graphs/corr_graphs/{}_{}_correlation_{}_noise.png".format(spectrum, choice_image, choice_map))
		except:
			os.mkdir("./graphs")
			plt.savefig("./encrypted_images/graphs/corr_graphs/{}_correlation_{}_noise.png".format(spectrum, choice_image, choice_map))
			
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
plain_img = cv2.imread("../{}.jpg".format(choice_image))
height, width, channels = plain_img.shape

n = height*width*channels

# Convert image into flat array
flat_array = plain_img.flatten()

# Initialize array for chaotic sequence
x = np.zeros(n)
x[0] = 0.1

# Initialize array for encrypted image
encrypted_array = np.zeros(n)

# Encrypt image pixel-wise using XOR operation
for i in range(1, n):
	
	encrypted_array[i-1] = flat_array[i-1] ^ int(x[i-1]*255)

	if choice_map == 'logistic':
		x[i] = r * x[i-1] * (1 - x[i-1])
		
	elif choice_map == 'tent':
		if x[i-1] < 0.5:
			x[i] = r * x[i-1]
		else:
			x[i] = r * (1-x[i-1])
		
	elif choice_map == 'sine':
		x[i] = (r * abs(math.sin(x[i-1]))) % 1 

# Reshape encrypted array according to original shape
encrypted_image = np.reshape(encrypted_array, (height, width, channels))

# Save file
try:
	cv2.imwrite("./encrypted_images/{}_encrypted_{}.png".format(choice_map, choice_image), encrypted_image)
except:
	os.mkdir("encrypted_images")
	cv2.imwrite("./encrypted_images/{}_encrypted_{}.png".format(choice_map, choice_image), encrypted_image)

# Ask if they want graphs to be plotted
choice_graphs = input("Do you want to plot graphs for this image? (Y/n) ")

if choice_graphs.casefold() == 'y':
	plot_hist_graph(encrypted_array)
	plot_corr_graphs(encrypted_image, plain_img)
