import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import os

# Function to plot histogram of image	
def plot_hist_graph(img, channel, index):
	
	# Convert image into float32 type
	img = np.float32(img)
	
	# Convert image to grayscale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	plt.hist(img.flatten(), bins=256, color='skyblue')
	plt.title("Histogram for {} Channel {}".format(channel.title(), choice_image.title()))
	plt.xlabel("Pixel intensity")
	plt.ylabel("Pixel frequency")
	
	try:
		os.mkdir("./graphs")
		os.mkdir("./graphs/histograms")
		plt.savefig("./graphs/histograms/{}_histogram_{}_channel.png".format(choice_image, channel))
	except:
		plt.savefig("./graphs/histograms/{}_histogram_{}_channel.png".format(choice_image, channel))
		
	plt.show()

# Function to plot correlation graph of image	
def plot_corr_graphs(img, channel, index):
	
	spectra = ['vertical', 'horizontal', 'diagonal']
	# Initialize empty numpy arrays to random pixel values
	a = np.zeros(4096)
	x = np.zeros(4096)
	y = np.zeros(4096)
	
	for i in range(4096):
		random_y = rand.randint(0, height-2)
		random_x = rand.randint(0, width-2)
		
		a[i] = img[random_y][random_x][index]
		
		if random_y+1 != height or random_x+1 != width:		# to ensure array does not exceed upper limit
			x[i] = img[random_y][random_x+1][index]		# for horizontal coefficient
			y[i] = img[random_y+1][random_x][index]		# for vertical coefficient
		
		elif random_y != 0 or random_x != 0:			# to ensure array does not exceed lower limit
			x[i] = img[random_y][random_x-1][index]	
			y[i] = img[random_y-1][random_x][index]	
		else:
			i = i-1		# To make it simple by starting over
	
	for spectrum in spectra:	# for all 3 spectra
	
		# Calculate correlation coefficient
		if spectrum == 'horizontal':
			corr_matrix = np.corrcoef(a, x)
			plt.scatter(a, x, s=5)
			
		elif spectrum == 'vertical':
			corr_matrix = np.corrcoef(a, y)
			plt.scatter(a, y, s=5)
			
		else:
			corr_matrix = np.corrcoef(x, y)
			plt.scatter(x, y, s=5)
		
		plt.title("Correlation Plot in {} spectrum for {} {} channel\nCorrelation coefficient = {}".format(spectrum.title(), choice_image.title(), channel.title(), corr_matrix[0][1]))
		
		try:
			os.mkdir("./graphs/corr_graphs")
			plt.savefig("./graphs/corr_graphs/{}_{}_correlation_{}_channel.png".format(spectrum, choice_image, channel))
		except:
			plt.savefig("./graphs/corr_graphs/{}_{}_correlation_{}_channel.png".format(spectrum, choice_image, channel))
			
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
		break
		
# Read image and its dimensions
original_image = cv2.imread("../{}.jpg".format(choice_image))
height, width, channels = original_image.shape

# Initialize array for extracting channels
img_array = np.zeros((height, width, channels))

# List of channels
channel_names = ['blue', 'green', 'red']

# Separate channels from image
for index, channel in enumerate(channel_names):
	
	# Copy pixel intensity of required channel into new array	
	img_array[:,:,index] = original_image[:,:,index]
	
	# Save image 
	try:
		os.mkdir("./separated_channels/")
		cv2.imwrite("./separated_channels/{}_channel_{}.jpg".format(channel, choice_image), img_array)
	except:	
		cv2.imwrite("./separated_channels/{}_channel_{}.jpg".format(channel, choice_image), img_array)
	
	# Ask if they want graphs to be plotted
	choice_graphs = input("Do you want to plot graphs for this image? (Y/n) ")

	if choice_graphs.casefold() == 'y':
		plot_hist_graph(img_array, channel, index)
		plot_corr_graphs(img_array, channel, index)
	
	# Reset array to zero
	img_array = np.zeros((height, width, channels))
	
