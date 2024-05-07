# Outlier Detection in Astronomical Images

This Jupyter notebook provides a method for detecting outliers in astronomical images using deep learning techniques. The notebook utilizes pre-trained EfficientNet models and nearest neighbors algorithm to identify images that deviate significantly from the majority.

## Setup
- Ensure you have the required libraries installed:
  - `torchvision` 0.13.1
  - `torch` 1.12.1
  - `numpy`  1.23.5
  - `matplotlib` 3.8.4
  - `astropy` 5.1
  - `pillow` 9.0.1
  - `efficientnet_pytorch` 0.7.1
  - `scikit-learn` 1.4.1

## Usage
1. **Data Preparation**:
   - Organize your astronomical image data in folders, one folder for each object.
   - It is important that the stacked image is the first in alphabetical order in your folder.
   - Update `rootpath` and `objpath` variables to point to the directory containing your image data.
   - Set the `size` parameter to the appropriate image size, we assume that the image is a square of `size`x`size` pixels.
   - `plot` is a boolean variable that enable or disable the plots
   - `timeStamp` is a boolean variable that enable or disable the print of timestamps

2. **Outlier Detection**:
   - Run the notebook cell by cell to execute the code.
   - The notebook will display detected outliers along with their respective images and filenames.


## Functions

- `fits_numpy`: Converts FITS format astronomical images to NumPy arrays.
- `normalize`: Normalizes image data.
- `NumpyDataset`: Custom dataset class for loading NumPy arrays.
- `get_latent_vectors`: Extracts latent vectors from images using a pre-trained EfficientNet model.
- `blockPrinting`: Decorator function to block printing output.
- `get_features`: Extracts features from images using EfficientNet model.
- `get_nns`: Finds nearest neighbors of a query image in the feature space.
- `searchOutliers`: Identifies outliers in the image dataset based on nearest neighbors distances.


## Note
- This notebook assumes the presence of GPU for efficient computation. Adjust the device configuration (`cuda:0` or `cpu`) based on your hardware availability.

For further details and updates, refer to the paper Cavuoti, De Cicco et al. arxiv:xxx
