import SimpleITK
import numpy as np
import os
from pandas.core.frame import DataFrame
from scipy.ndimage.interpolation import zoom


def load_itk(mhd_filepath: str) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function reads a '.mhd' file using SimpleITK and return the image array,
    origin and spacing of the image.
    :param mhd_filepath:
    :return:
    """

    # Reads the image using SimpleITK
    itk_image = SimpleITK.ReadImage(mhd_filepath)

    # Convert the image to a  numpy array first and then shuffle the
    # dimensions to get axis in the order z,y,x
    ct_scan = SimpleITK.GetArrayFromImage(itk_image)

    # Read the origin of the ct_scan, will be used to convert the coordinates
    # from world to voxel and vice versa.
    origin = np.array(list(reversed(itk_image.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itk_image.GetSpacing())))

    return ct_scan, origin, spacing


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return [start + step * i for i in range(n + 1)]
    else:
        return []


def world_2_voxel(world_coordinates, origin, spacing):
    """
    This function is used to convert the world coordinates to voxel coordinates using 
    the origin and spacing of the ct_scan
    """

    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


def draw_circles(image: np.ndarray, candidates: DataFrame, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    This function is used to create spherical regions in binary masks
    at the given locations and radius.
    """

    # make empty matrix, which will be filled with the mask
    resize_spacing = [1, 1, 1]
    image_mask = np.zeros(image.shape)

    # run over all the nodules in the lungs
    for candidate in candidates.values:
        # get middle x-,y-, and z-world_coordinate of the nodule
        radius = np.ceil(candidate[4]) / 2
        coord_x = candidate[1]
        coord_y = candidate[2]
        coord_z = candidate[3]
        image_coord = np.array((coord_z, coord_y, coord_x))

        # determine voxel coordinate given the world_coordinate
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        nodule_range = seq(-radius, radius, resize_spacing[0])

        # create the mask
        for x in nodule_range:
            for y in nodule_range:
                for z in nodule_range:
                    world_coordinates = np.array((coord_z + z, coord_y + y, coord_x + x))
                    coordinates = world_2_voxel(world_coordinates, origin, spacing)
                    if (np.linalg.norm(image_coord - coordinates) * resize_spacing[0]) < radius:
                        rx = np.round(coordinates[0])
                        ry = np.round(coordinates[1])
                        rz = np.round(coordinates[2])
                        image_mask[rx, ry, rz] = int(1)
    return image_mask


def resample(img: np.ndarray, spacing: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    resize images to 1mm scale
    :param img: 3D image tensor
    :param spacing:
    :return: 
    """
    # calculate resize factor
    resize_spacing = [1, 1, 1]
    resize_factor = spacing / resize_spacing
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    return zoom(img, real_resize, mode='nearest'), new_spacing


def images_and_nodules(patient_id: str, annotations: DataFrame, image_dir: str) -> (np.ndarray, np.ndarray):
    """
    extract images from mhd/raw images and create nodule masks from the annotations
    :param patient_id: unique id which is called seriesuid
    :param annotations: nodule annotation DataFrame
    :param image_dir: directory which has CT scan images
    :return: 3D image array and mask of nodules
    """

    mhd_filepath = "{}/{}.mhd".format(image_dir, patient_id)
    if not os.path.exists(mhd_filepath):
        return None

    img, origin, spacing = load_itk(mhd_filepath)
    candidates = annotations[annotations["seriesuid"] == patient_id]

    lung_img, new_spacing = resample(img, spacing)
    nodule_mask = draw_circles(lung_img, candidates, origin, new_spacing)
    return lung_img, nodule_mask
