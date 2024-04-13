# SVD Image Compression Tool

This project provides a Python script with a graphical user interface (GUI) for compressing images using Singular Value Decomposition (SVD). It allows users to select an input image, choose the desired output directory, and perform the compression operation, which results in various levels of compressed images and a plot of error versus compression ratio.

## Requirements

Before running this script, make sure the following Python packages are installed:

- NumPy
- Matplotlib
- PIL (Pillow)
- Tkinter (usually included by default in Python installations)

You can install the necessary packages using pip:

```bash
pip install numpy matplotlib pillow
```

## Usage

To use the tool, follow these steps:

1. Run the script via command line or an IDE.
2. The GUI window will open.
3. Click on "Browse Image" to select the image you want to compress. If you cannot find the image file in the selected folder, please make sure you select "All files" option.
4. Click on "Browse Output Folder" to select the destination for the compressed images and additional generated files.
5. Click on "Compress Image" to start the compression process.

After the compression is finished, you will see:

- Compressed versions of the original image at various compression levels.
- A plot image showing the error vs. compression ratio.
- A text file summarizing the compression errors and ratios.

## Output Description

The tool generates:

- A PNG image for each specified compression percentage with labels indicating the compression percentage.
- A PNG image showing a plot of error versus compression ratio.
- A text file (`error_compression_data.txt`) containing a list of each percentage with corresponding error and compression ratio.

These files will be saved in the selected output directory.

## Features

- GUI for easy interaction.
- Several levels of image compression based on different percentages of singular value retention.
- Calculation and visualization of error vs. compression ratio.
- Handling of PNG and JPEG image files for compression.
