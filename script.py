import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from svd_class import svd_2



def svd_image_compression(input_image_path, output_dir, custom_svd=True):
    original_image = Image.open(input_image_path)
    image_array = np.array(original_image)

    # Process the image to get the 3 matrices R, G, B
    R = image_array[:, :, 0]
    G = image_array[:, :, 1]
    B = image_array[:, :, 2]

    # Process to get the original rank of the image
    R_rank = np.linalg.matrix_rank(R)
    G_rank = np.linalg.matrix_rank(G)
    B_rank = np.linalg.matrix_rank(B)

    # Percentages for compression
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
        if custom_svd:
            U_R, s_R, VT_R = svd_2(R, 10000000, 1e-12)()
            U_G, s_G, VT_G = svd_2(G, 10000000, 1e-12)()
            U_B, s_B, VT_B = svd_2(B, 10000000, 1e-12)()
        else:
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

        # Calculate the compression ratio
        image_size = np.prod(image_array.shape)
        compressed_size = (rank['R'] * (U_R.shape[1] + VT_R.shape[0] + 1) +
                           rank['G'] * (U_G.shape[1] + VT_G.shape[0] + 1) +
                           rank['B'] * (U_B.shape[1] + VT_B.shape[0] + 1))
        compression_ratio = compressed_size / image_size
        compression_ratios.append(compression_ratio)

        # Calculate the error
        error = np.linalg.norm(image_array - compressed_image)
        errors.append(error)

        # Output the compressed image
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        compressed_image_path = os.path.join(output_dir, 'compressed_image_{}.png'.format(int(percentage * 100)))
        Image.fromarray(compressed_image.astype(np.uint8)).save(compressed_image_path)

    # Plot the error vs compression ratio
    plt.figure(figsize=(10, 6))
    plt.plot(compression_ratios, errors, marker='o')
    plt.title('Error vs Compression Ratio')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_vs_compression_ratio.png'))
    plt.close()

    # Output the error and compression ratio
    error_compression_output_path = os.path.join(output_dir, 'error_compression_data.txt')
    with open(error_compression_output_path, 'w') as f:
        for percentage, error, ratio in zip(percentages, errors, compression_ratios):
            ratio_percentage = ratio * 100  # Convert ratio to percentage
            f.write('Percentage: {:.0f}%, Error: {:.2f}, Compression Ratio: {:.2f}%\n'.format(percentage * 100, error,
                                                                                              ratio_percentage))


def open_file_dialog():
    file_path = filedialog.askopenfilename(
        filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        image_path.set(file_path)
        tickle_image_label.config(text='✔️ Image Selected', fg='green')


def open_output_directory_dialog():
    directory_path = filedialog.askdirectory()
    if directory_path:
        output_path.set(directory_path)
        tickle_output_label.config(text='✔️ Output Folder Selected', fg='green')


def run_process():
    input_image_path = image_path.get()
    selected_output_dir = output_path.get()

    if not input_image_path:
        messagebox.showwarning("Warning", "Please select an image file.")
        return
    if not selected_output_dir:
        messagebox.showwarning("Warning", "Please select an output folder.")
        return

    try:
        svd_image_compression(input_image_path, selected_output_dir)
        messagebox.showinfo("Success",
                            f"SVD image compression complete. Compressed images and error vs compression ratio are saved in: {selected_output_dir}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


root = tk.Tk()
root.title('SVD Image Compression GUI')

my_font = ('Helvetica', 20)

image_path = tk.StringVar()
output_path = tk.StringVar()

tk.Label(root, text='Select an image to compress:', font=my_font).pack(padx=10, pady=5)
tk.Button(root, text="Browse Image", command=open_file_dialog, font=my_font).pack(padx=10, pady=5)
tickle_image_label = tk.Label(root, text='', font=my_font)
tickle_image_label.pack()

tk.Label(root, text='Select output folder:', font=my_font).pack(padx=10, pady=5)
tk.Button(root, text="Browse Output Folder", command=open_output_directory_dialog, font=my_font).pack(padx=10, pady=5)
tickle_output_label = tk.Label(root, text='', font=my_font)
tickle_output_label.pack()
tk.Button(root, text="Compress Image", command=run_process, font=my_font).pack(padx=10, pady=20)

root.mainloop()