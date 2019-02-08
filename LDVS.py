import numpy as np
from osgeo import gdal
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
    
    image = np.transpose(gdal.Open(input_MSfile).ReadAsArray(), [1, 2, 0])
    gray = np.zeros((int(image.shape[0]), int(image.shape[1])))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray[i, j] = max(image[i, j, 0], image[i, j, 1], image[i, j, 2])
    
    input_dataset = gdal.Open(input_MSfile)
    input_band = input_dataset.GetRasterBand(1)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff_driver.Create(output_filename, input_band.XSize, input_band.YSize, 1, gdal.GDT_Float32)
    output_dataset.SetProjection(input_dataset.GetProjection())
    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.GetRasterBand(1).WriteArray(gray)
    
    output_dataset.FlushCache()
    del output_dataset
    
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
    - step_size: Threshold increment for LDVS calculation
    - write: Boolean indicating whether to write LDVS feature map to file
    
    Outputs:
    - LDVS_rescaled: LDVS feature map normalized to range [0, 1] as implemented in the paper
    
    """
    
    if (window % 2 == 0) :
        raise ValueError('window size must be an odd number.')
    else :    
        buffer = int((window - 1) / 2) 
    
    pan_image = gdal.Open(input_grayfile).ReadAsArray()
    intensity_image = np.hypot(ndimage.sobel(pan_image, 0), ndimage.sobel(pan_image, 1))
    
    LDVS_image = np.zeros((intensity_image.shape[0], intensity_image.shape[1]))
    
    for thresh in tqdm(range(LB, UB + 1, step_size), mininterval = 60):
        thresholded_image = intensity_image * (intensity_image > np.percentile(intensity_image, thresh))
    
        for alpha in range(buffer, thresholded_image.shape[0] - buffer):            
            for beta in range(buffer, thresholded_image.shape[1] - buffer):                                                                                                                                   
                array = thresholded_image[(alpha - buffer) : (alpha + buffer + 1), (beta - buffer) : (beta + buffer + 1)]
                LDVS_image[alpha, beta] += np.amin(array)
                
    LDVS_rescaled = exposure.rescale_intensity(LDVS_image, out_range = (0, 1))
                    
    if write:
        input_dataset = gdal.Open(input_grayfile)
        input_band = input_dataset.GetRasterBand(1)
        gtiff_driver = gdal.GetDriverByName('GTiff')
        output_dataset = gtiff_driver.Create(output_ldvs_name, input_band.XSize, input_band.YSize, 1, gdal.GDT_UInt16)
        output_dataset.SetProjection(input_dataset.GetProjection())
        output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
        output_dataset.GetRasterBand(1).WriteArray(LDVS_rescaled)
    
        output_dataset.FlushCache()
        del output_dataset
    
    return LDVS_rescaled
