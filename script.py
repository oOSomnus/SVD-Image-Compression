# SVD Image Compression Script

# 1. Importing Libraries
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
from skimage import color

# 2. Importing Image
image_path = './input/42049.jpg'  # modify with your image path
original_image = Image.open(image_path)

# 3. Process the image to get the 3 matrices
image_array = np.array(original_image)
# If the image is RGB, we can convert it to grayscale, or separately process each channel.
# Here we treat the image as grayscale for simplicity.
if len(image_array.shape) == 3:
    image_array = color.rgb2gray(image_array)

# 4. Process to get the original rank of the image
U, s, VT = np.linalg.svd(image_array, full_matrices=False)
original_rank = np.linalg.matrix_rank(image_array)

# 5. For 5%, 10%, 30%, 60%, 80%, and 100% of the original rank, calculate the compressed image with the SVD
percentages = [0.05, 0.1, 0.3, 0.6, 0.8, 1.0]
compression_ratios = []
errors = []

for percentage in percentages:
    rank = int(original_rank * percentage)
    compressed_image = (U[:, :rank] * s[:rank]) @ VT[:rank, :]
    
    # 6. Calculate the compression ratio
    image_size = image_array.shape[0] * image_array.shape[1]
    compressed_size = rank * (U.shape[1] + VT.shape[0] + 1)
    compression_ratio = compressed_size / image_size
    compression_ratios.append(compression_ratio)
    
    # 7. Calculate the error
    error = np.linalg.norm(image_array - compressed_image)
    errors.append(error)
    
    # 9. Output the compressed image
    output_dir = './output'  # You can change this target directory if required
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    compressed_image_path = os.path.join(output_dir, 'compressed_image_{}.png'.format(int(percentage * 100)))
    Image.fromarray(np.uint8(compressed_image * 255)).save(compressed_image_path)

# 8. Plot the error vs compression ratio
plt.figure(figsize=(10, 6))
plt.plot(compression_ratios, errors, marker='o')
plt.title('Error vs Compression Ratio')
plt.xlabel('Compression Ratio')
plt.ylabel('Error')
plt.grid(True)
plt.savefig('error_vs_compression_ratio.png')
plt.show()

# Output the error and compression ratio
error_compression_output_path = 'error_compression_data.txt'
with open(error_compression_output_path, 'w') as f:
    for percentage, error, ratio in zip(percentages, errors, compression_ratios):
        f.write('Percentage: {:.0f}%, Error: {:.2f}, Compression Ratio: {:.2f}\n'.format(percentage * 100, error, ratio))

print('SVD image compression complete. Compressed images and graphs have been saved.')