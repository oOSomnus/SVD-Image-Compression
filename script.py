import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Importing Libraries
from scipy.linalg import svd

# 2. Importing Image
image = Image.open('image.jpg')
image = image.convert('L')  # convert image to grayscale
data = np.array(image)

# 3. Process the image to get the 3 matrices
U, S, V = svd(data)

# 4. Process to get the original rank of the image
original_rank = np.linalg.matrix_rank(data)

# 5. For 5%, 10%, 30%, 60% and 90% of the original rank, calculate the compressed image with the SVD
ranks = [int(original_rank * x / 100) for x in [5, 10, 30, 60, 90]]
compressed_images = []
for rank in ranks:
    compressed_image = np.dot(U[:, :rank], np.dot(np.diag(S[:rank]), V[:rank, :]))
    compressed_images.append(compressed_image)

# 6. Calculate the compression ratio
compression_ratios = [original_rank / rank for rank in ranks]

# 7. Calculate the error
errors = [np.linalg.norm(data - compressed_image) for compressed_image in compressed_images]

# 8. Plot the error vs compression ratio
plt.plot(compression_ratios, errors)
plt.xlabel('Compression Ratio')
plt.ylabel('Error')
plt.show()

# 9. Output the compressed image, the error and the compression ratio, and the plot
for i in range(len(ranks)):
    print(f'Compression Ratio: {compression_ratios[i]}, Error: {errors[i]}')
    Image.fromarray(compressed_images[i]).show()