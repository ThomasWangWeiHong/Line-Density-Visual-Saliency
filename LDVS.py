import numpy as np
import rasterio
from scipy import ndimage
from skimage import exposure
from tqdm import tqdm



def grayscale_raster_creation(input_MSfile, output_filename):
    """ 
    This function creates a grayscale brightness image from an input image to be used for PanTex index calculation. For every pixel 
    in the input image, the intensity values from the red, green, blue channels are first obtained, and the maximum of these values 
    are then assigned as the pixel's intensity value, which would give the grayscale brightness image as mentioned earlier, as per 
    standard practice in the remote sensing academia and industry. It is assumed that the first three channels of the input image 
    correspond to the red, green and blue channels, irrespective of order.
    
    Inputs:
    - input_MSfile: File path of the input image that needs to be converted to grayscale brightness image
    - output_filename: File path of the grayscale brightness image that is to be written to file
    
    Outputs:
    - gray: Numpy array of grayscale brightness image of corresponding multi - channel input image
    
    """
    
    with rasterio.open(input_MSfile) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])[:, :, 0 : 3]
        
    gray = np.max(img, axis = 2)
    
    metadata['count'] = 1
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(gray[np.newaxis, :, :])
    
    return gray



def LDVS_feature_map_creation(input_grayfile, output_ldvs_name, window = 3, LB = 20, UB = 100, step_size = 20, write = True):
    """ 
    This function is used to create a Line Density Visual Saliency (LDVS) feature map using a Sobel filter, as implemented in 
    the paper 'Automatic Newly Increased Built - Up Area Extraction From High - Resolution Remote Sensing Images Using 
    Line - Density - Based Visual Saliency and PanTex' by Wu T., Luo J., Zhou X., Ma J., Song X. (2018). 
    
    Inputs:
    - input_grayfile: File path of input grayscale image to be used
    - output_ldvs_name: File path of output LDVS feature map
    - window: Size of window to be used for calculation of LDVS value for a pixel
    - LB: Lower bound of histogram of edge intensity map for thresholding
    - UB: Upper bound of histogram of edge intensity map for thresholding
    - step_size: Percentile increment in threshold for LDVS calculation
    - write: Boolean indicating whether to write LDVS feature map to file
    
    Outputs:
    - LDVS_rescaled: LDVS feature map normalized to range [0, 1] as implemented in the paper
    
    """
    
    if (window % 2 == 0) :
        raise ValueError('window size must be an odd number.')
    else :    
        buffer = int((window - 1) / 2) 
    
    with rasterio.open(input_grayfile) as f:
        metadata = f.profile
        pan_image = f.read(1)
        
    
    intensity_image = np.hypot(ndimage.sobel(pan_image, 0), ndimage.sobel(pan_image, 1))
    
    LDVS_image = np.zeros((intensity_image.shape[0], intensity_image.shape[1]))
    
    for thresh in tqdm(range(LB, UB + 1, step_size), mininterval = 60):
        thresholded_image = intensity_image * (intensity_image > np.percentile(intensity_image, thresh))
    
        for alpha in range(buffer, thresholded_image.shape[0] - buffer):            
            for beta in range(buffer, thresholded_image.shape[1] - buffer):                                                                                                                                   
                array = thresholded_image[(alpha - buffer) : (alpha + buffer + 1), (beta - buffer) : (beta + buffer + 1)]
                LDVS_image[alpha, beta] += np.amin(array)
                
    LDVS_rescaled = exposure.rescale_intensity(LDVS_image, out_range = (0, 1)).astype(np.float32)
                    
    if write:
        metadata['dtype'] = 'float32'
        with rasterio.open(output_ldvs_name, 'w', **metadata) as dst:
            dst.write(LDVS_rescaled[np.newaxis, :, :])
    
    return LDVS_rescaled
