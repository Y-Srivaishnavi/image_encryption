import cv2
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import os

# Function to put noise in image
def put_noise_in_image(original_array, amount_of_noise):
	img_array = np.copy(original_array)
	
	height, width = img_array.shape
	number_of_pixels = height*width*(amount_of_noise/100)
	
	# Add 'peppered' noise i.e. convert image pixels to black
	for pepper in range(int(number_of_pixels/2)):
		random_y = rand.randint(0, height-1)
		random_x = rand.randint(0, width-1)
		
		img_array[random_y][random_x] = 0
	
	# Add 'salted' noise i.e. convert image pixels to white
	for salt in range(int(number_of_pixels/2)):
		random_y = rand.randint(0, height-1)
		random_x = rand.randint(0, width-1)
		
		img_array[random_y][random_x] = 255
		
	return img_array

# Function to plot histogram of image	
def plot_hist_graph(img_with_noise):
	
	plt.hist(img_with_noise.flatten(), bins=256, color='skyblue')
	plt.title("Histogram for {} with {}% noise".format(choice_image.title(), choice_noise))
	plt.xlabel("Pixel intensity")
	plt.ylabel("Pixel frequency")
	
	try:
		os.mkdir("./graphs/")
		os.mkdir("./graphs/histograms")
		plt.savefig("./graphs/histograms/{}_histogram_{}_noise.png".format(choice_image, choice_noise))
	
	except:
		plt.savefig("./graphs/histograms/{}_histogram_{}_noise.png".format(choice_image, choice_noise))
		
	plt.show()

# Function to plot correlation graph of image	
def plot_corr_graphs(img_with_noise, img_without_noise):
	
	spectra = ['vertical', 'horizontal', 'diagonal']
	height, width = img_with_noise.shape
	
	# Initialize empty numpy arrays to random pixel values
	a = np.zeros(4096)
	x = np.zeros(4096)
	y = np.zeros(4096)
	z = np.zeros(4096)
	
	for i in range(4096):
		random_y = rand.randint(0, height-2)
		random_x = rand.randint(0, width-2)
		
		a[i] = img_with_noise[random_y][random_x]
		
		if random_y+1 != height or random_x+1 != width:		# to ensure array does not exceed upper limit
			x[i] = img_without_noise[random_y][random_x+1]		# for horizontal coefficient
			y[i] = img_without_noise[random_y+1][random_x]		# for vertical coefficient
			z[i] = img_without_noise[random_y+1][random_x+1]	# for diagonal coefficient
		
		elif random_y != 0 or random_x != 0:			# to ensure array does not exceed lower limit
			x[i] = img_without_noise[random_y][random_x-1]	
			y[i] = img_without_noise[random_y-1][random_x]	
			z[i] = img_without_noise[random_y-1][random_x-1]	
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
		plt.title("Correlation Plot in {} spectrum for {} with {}% noise\nCorrelation coefficient = {}".format(spectrum.title(), choice_image.title(), choice_noise, corr_matrix[0][1]))
		plt.xlabel("Image without noise")
		plt.ylabel("Image with noise")
		
		try:
			os.mkdir("./graphs/corr_plots")
			plt.savefig("./graphs/corr_graphs/{}_{}_correlation_{}_noise.png".format(spectrum, choice_image, choice_noise))
		except:
			plt.savefig("./graphs/corr_graphs/{}_correlation_{}_noise.png".format(spectrum, choice_image, choice_noise))
			
		plt.show()

# Select image and amount of noise to be put
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
		
	choice_noise = int(input("How much percent of noise should be put? "))
	break

# Read image
original_image = cv2.imread("../{}.jpg".format(choice_image))

# Convert image to grayscale
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Put noise in image
noisy_image = put_noise_in_image(grayscale_image, choice_noise)

# Save image 
try:
	os.mkdir("./images/")
	cv2.imwrite("./images/{}_with_noise_{}_percent.jpg".format(choice_image, choice_noise), noisy_image)

except:
	cv2.imwrite("./images/{}_with_noise_{}_percent.jpg".format(choice_image, choice_noise), noisy_image)

# Ask if they want graphs to be plotted
choice_graphs = input("Do you want to plot graphs for this image? (Y/n) ")

if choice_graphs.casefold() == 'y':
	plot_hist_graph(noisy_image)
	plot_corr_graphs(noisy_image, grayscale_image)

