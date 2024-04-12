# SVD Image Compression Script

# 1. Importing Libraries
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image

# 2. Importing Image
image_path = './input/42049.jpg'
original_image = Image.open(image_path)
image_array = np.array(original_image)

# 3. Process the image to get the 3 matrices R, G, B
R = image_array[:, :, 0]
G = image_array[:, :, 1]
B = image_array[:, :, 2]

# 4. Process to get the original rank of the image
# (Assuming all channels have the same dimensions, hence the same rank)
R_rank = np.linalg.matrix_rank(R)
G_rank = np.linalg.matrix_rank(G)
B_rank = np.linalg.matrix_rank(B)

# 5. For 5%, 10%, 30%, 60%, 80%, and 100% of the original rank, calculate the compressed image with the SVD
percentages = [0.05, 0.1, 0.3, 0.6, 0.8, 1.0]
compression_ratios = []
errors = []

for percentage in percentages:
    rank = {
        'R': int(R_rank * percentage),
        'G': int(G_rank * percentage),
        'B': int(B_rank * percentage)
    }
    
    # Apply SVD for each channel
    U_R, s_R, VT_R = np.linalg.svd(R, full_matrices=False)
    U_G, s_G, VT_G = np.linalg.svd(G, full_matrices=False)
    U_B, s_B, VT_B = np.linalg.svd(B, full_matrices=False)
    
    # Reconstruct the image channels at the given rank
    R_compressed = (U_R[:, :rank['R']] * s_R[:rank['R']]) @ VT_R[:rank['R'], :]
    G_compressed = (U_G[:, :rank['G']] * s_G[:rank['G']]) @ VT_G[:rank['G'], :]
    B_compressed = (U_B[:, :rank['B']] * s_B[:rank['B']]) @ VT_B[:rank['B'], :]
    
    # Stack the channels back together
    compressed_image = np.stack((R_compressed, G_compressed, B_compressed), axis=-1)
    compressed_image = np.clip(compressed_image, 0, 255)  # Clipping to valid range
    
    # 6. Calculate the compression ratio
    image_size = np.prod(image_array.shape)
    compressed_size = (rank['R'] * (U_R.shape[1] + VT_R.shape[0] + 1) +
                       rank['G'] * (U_G.shape[1] + VT_G.shape[0] + 1) +
                       rank['B'] * (U_B.shape[1] + VT_B.shape[0] + 1))
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
    Image.fromarray(compressed_image.astype(np.uint8)).save(compressed_image_path)

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
        f.write('Percentage: {:.0f}%, Error: {:.2f}, Compression Ratio: {:.4f}\n'.format(percentage * 100, error, ratio))

print('SVD image compression complete. Compressed images and graphs have been saved.')