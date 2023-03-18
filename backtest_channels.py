import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import os

# Function to plot histogram of image	
def plot_hist_graph(backtested_image):
	
	# Convert images to int type
	backtested_image =  np.float32(backtested_image)
	
	# Read image as grayscale
	backtested_image = cv2.cvtColor(backtested_image, cv2.COLOR_BGR2GRAY)
	
	plt.hist(backtested_image.flatten(), bins=256, color='skyblue')
	plt.title("Histogram for {}".format(choice_image.title()))
	plt.xlabel("Pixel intensity")
	plt.ylabel("Pixel frequency")
	
	try:
		os.mkdir("./backtested/graphs/")
		os.mkdir("./backtested/graphs/histograms")	
		plt.savefig("./backtested/graphs/histograms/test_{}_histogram.png".format(choice_image))
	
	except:
		plt.savefig("./backtested/graphs/histograms/test_{}_histogram.png".format(choice_image))
	
	plt.show()
	
# Function to plot correlation graph of image	
def plot_corr_graphs(backtested_img, original_img):
	
	# Convert images to int type
	backtested_img =  np.float32(backtested_img)
	
	# Read images as grayscale
	backtested_img = cv2.cvtColor(backtested_img, cv2.COLOR_BGR2GRAY)
	original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	
	spectra = ['vertical', 'horizontal', 'diagonal']
	height, width = backtested_img.shape
	
	# Initialize empty numpy arrays to random pixel values
	a = np.zeros(4096)
	x = np.zeros(4096)
	y = np.zeros(4096)
	z = np.zeros(4096)
	
	for i in range(4096):
		random_y = rand.randint(0, height-2)
		random_x = rand.randint(0, width-2)
		
		a[i] = backtested_img[random_y][random_x]
		
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
		plt.scatter(a, temp_array, s=5)
		plt.title("Correlation Plot for {}\nCorrelation coefficient = {}".format(choice_image.title(), corr_matrix[0][1]))
		plt.xlabel("Backtested Image")
		plt.ylabel("Original Image")
		
		try:
			os.mkdir("./backtested/graphs/corr_graphs")
			plt.savefig("./backtested/graphs/corr_graphs/test_{}_correlation.png".format(choice_image))
		except:
			plt.savefig("./backtested/graphs/corr_graphs/test_{}_correlation.png".format(choice_image))
			
		plt.show()
		
# Select image 
while True: 

	print("Select one image: ")
	print("1. Lena")
	print("2. Baboon")
	print("3. Peppers")
	choice_image = input("Your choice: ")
	
	if choice_image == '1' or choice_image.casefold() in 'lena':
		choice_image = 'lena'
		break
		
	elif choice_image == '2' or choice_image.casefold() in 'baboon':
		choice_image = 'baboon'
		break
		
	elif choice_image == '3' or choice_image.casefold() in 'peppers':
		choice_image = 'peppers'
		break
		
	else:
		print("Invalid choice, try again.\n")

# Read original image
original_image = cv2.imread("../{}.jpg".format(choice_image))

# Note dimensions of image
height, width, channels = original_image.shape

# Initialize array of new image
test_image = np.zeros((height, width, channels))

# List of channels
channel_names = ['blue', 'green', 'red']

# Separate channels from image
for index, channel in enumerate(channel_names):
	
	# Read image channels
	temp_channel = cv2.imread("./separated_channels/{}_channel_{}.jpg".format(channel, choice_image))
	
	# Copy pixel intensity of required channel into new array	
	test_image[:,:,index] = temp_channel[:,:,index]

# Save image 
try:
	os.mkdir("./backtested/")
	cv2.imwrite("./backtested/test_{}.jpg".format(choice_image), test_image)
except:	
	cv2.imwrite("./backtested/test_{}.jpg".format(choice_image), test_image)


# Ask if they want graphs to be plotted
choice_graphs = input("Do you want to plot graphs for this image? (Y/n) ")

if choice_graphs.casefold() == 'y':
	plot_hist_graph(test_image)
	plot_corr_graphs(test_image, original_image)
	
