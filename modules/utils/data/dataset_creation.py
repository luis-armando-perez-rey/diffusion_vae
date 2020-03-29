import numpy as np
import itertools
import glob
import imageio
import os
import PIL
import json


def flatten_normalize_images(images):
    image_size = np.product(images.shape[1:])
    x_train = np.reshape(images, [-1, image_size])
    x_train = x_train.astype('float32') / np.amax(x_train)
    return x_train


def normalize_images(images):
    images = images.astype('float32') / np.amax(images)
    return images


def binarize(data, seed):
    assert np.amax(data) <= 1.0 and np.amin(data) >= 0.0, "Values not normalized"
    np.random.seed(seed)
    binarized = (np.random.uniform(0.0, 1.0, data.shape) < data).astype(int)
    return binarized


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def circular_shift_image(image: np.ndarray, w_pix_shift: int, h_pix_shift: int):
    """
        Takes a numpy array and shifts it it periodically along the
        width and the height
        :param image: input image, must be at least a 2D numpy array
        :param w_pix_shift: number of pixels that the image is shifted in the width
        :param h_pix_shift: number of pixels that the image is shifted in the height
        :return: numpy array of the shifted image
        """
    # Shift the image along the width
    shifted_image = np.roll(image, w_pix_shift, axis=1)
    # Shift the image along the height
    shifted_image = np.roll(shifted_image, h_pix_shift, axis=0)
    return shifted_image


def combinations_circular_shift(image: np.ndarray):
    (height, width, channels) = image.shape
    shifted_images = np.zeros((height * width, height, width, channels))
    shifts = np.zeros((height * width, 2))
    for i in range(height):
        for j in range(width):
            shifted_images[i * width + j] = circular_shift_image(image, i, j)
            shifts[i * width + j, 0] = i
            shifts[i * width + j, 1] = j
    return shifts, shifted_images


def sinsuoid_image_random(num_samples, n_T, omega_values):
    phases = np.random.uniform(0, 1, (num_samples, 2)) * 2 * np.pi
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    sinusoid_images = np.zeros((n_T, n_T, len(phases)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phases), axis=2)
        sinusoid_images += np.sin(np.add(repeated_volume, phases[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phases, sinusoid_images


def uniform_component_sineimage(n_T, num_components):
    num_samples = 1000
    space_linspace = np.linspace(0, 1, n_T)
    phases = np.random.uniform(0, 1, (num_samples, 2))
    # Omega Volume
    components = np.array(range(-num_components, num_components + 1, 1))
    omega_combinations = []
    for i, j in itertools.product(components, components):
        omega_combinations.append((i, j))
    spatial_meshes = np.meshgrid(space_linspace, space_linspace)
    omega_combinations = np.array(omega_combinations)

    combinations_volume = np.ones((n_T, n_T, len(omega_combinations), num_samples)) + 0j

    for i in range(2):
        omega_volumex = omega_combinations[:, i, np.newaxis, np.newaxis, np.newaxis]
        omega_volumex = np.repeat(omega_volumex, repeats=n_T, axis=1)
        omega_volumex = np.repeat(omega_volumex, repeats=n_T, axis=2)
        omega_volumex = np.repeat(omega_volumex, repeats=num_samples, axis=3)
        omega_volumex = np.swapaxes(omega_volumex, 0, 2)

        spatial_mesh = spatial_meshes[i]
        mesh_expanded_dim = spatial_mesh[:, :, np.newaxis, np.newaxis]
        mesh_expanded_dim = np.repeat(mesh_expanded_dim, repeats=len(omega_combinations), axis=2)
        mesh_expanded_dim = mesh_expanded_dim - 0.5
        mesh_expanded_dim = np.repeat(mesh_expanded_dim, repeats=num_samples, axis=3)

        phases_volumex = phases[:, i, np.newaxis, np.newaxis, np.newaxis]
        phases_volumex = np.repeat(phases_volumex, repeats=n_T, axis=1)
        phases_volumex = np.repeat(phases_volumex, repeats=n_T, axis=3)
        phases_volumex = np.repeat(phases_volumex, repeats=len(omega_combinations), axis=2)
        phases_volumex = np.swapaxes(phases_volumex, 0, 3)

        combinations_volume *= np.exp(2 * omega_volumex * np.pi * (mesh_expanded_dim + phases_volumex) * 1j)
    combinations = np.sum(combinations_volume, axis=2).real
    combinations = np.swapaxes(combinations, 2, 0)
    combinations = combinations / np.amax(combinations)
    return phases, combinations


def sinusoid_image_phase_combination(num_samples1, num_samples2, n_T, omega_values):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """

    phases1 = np.linspace(0, 1, num_samples1) * 2 * np.pi
    phases2 = np.linspace(0, 1, num_samples2) * 2 * np.pi
    # Sampling from phase and space
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    phase_combinations = np.array(list(itertools.product(phases1, phases2)))
    sinusoid_images = np.zeros((n_T, n_T, len(phase_combinations)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phase_combinations), axis=2)
        sinusoid_images += np.sin(np.add(repeated_volume, phase_combinations[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phase_combinations, sinusoid_images


def random_so3_matrices(num_samples: int):
    """
    Creates num_samples random 3x3 rotation matrices from SO(3) manifold
    :param num_samples (int): number of rotation matrices to be created
    :return: returns an array with matrices together with the corresponding Euler angles
    """

    # Random initial matrix
    random_matrices = np.random.normal(0.0, 1.0, (num_samples, 3, 3))
    # Matrix decomposition
    u, s, vh = np.linalg.svd(random_matrices)
    # Orthogonal matrix
    orthogonal_matrices = np.matmul(u, vh)
    so3_matrices = np.copy(orthogonal_matrices)
    angles = np.zeros((num_samples, 3))
    for num_matrix, matrix in enumerate(orthogonal_matrices):
        # Make the determinant positive
        so3_matrices[num_matrix] = np.linalg.det(matrix) * matrix
        # Calculate corresponding angle
        angles[num_matrix, :] = rotationMatrixToEulerAngles(so3_matrices[num_matrix])
    return so3_matrices, angles


def random_o3_matrices(num_samples: int):
    """
    Creates num_samples random 3x3 matrices from O(3) manifold
    :param num_samples: number of matrices to be created
    :return: returns an array with matrices together with the corresponding Euler angles of the corresponding SO(3)
    manifold
    """
    # Random initial matrix
    random_matrices = np.random.normal(0.0, 1.0, (num_samples, 3, 3))
    # Matrix decomposition
    u, s, vh = np.linalg.svd(random_matrices)
    # Orthogonal matrix
    orthogonal_matrices = np.matmul(u, vh)
    angles = np.zeros((num_samples, 3))
    for num_matrix, matrix in enumerate(orthogonal_matrices):
        # Calculate corresponding angle from SO(3) matrix
        angles[num_matrix, :] = rotationMatrixToEulerAngles(np.linalg.det(matrix) * matrix)
    return orthogonal_matrices, angles


def flatten_matrix(matrices):
    """
    Flatten an array of 3x3 matrices
    :param matrices: Input matrices
    :return: Output flattened matrices array
    """
    flat_matrices = np.reshape(matrices, (-1, 9))
    return flat_matrices


def select_supervision(labels, percentage_kept, canonical=True):
    if canonical:
        np.random.seed(17)

    # Supervision
    supervision = np.ones(len(labels), dtype=float)
    num_labelled_images = int(np.round((1 - percentage_kept) * len(labels)))
    labels_indexes = np.arange(len(labels))
    np.random.shuffle(labels_indexes)
    erase_labels_index = labels_indexes[:num_labelled_images]
    labels[erase_labels_index, :] = np.zeros(shape=labels[erase_labels_index, :].shape)
    supervision[erase_labels_index] = 0.0
    np.random.seed(None)
    return labels, supervision

def select_supervision_object_based(labels, object_identifiers, percentage_kept, canonical=True):
    """
    This function produces the labels and the supervision array for training the semi-supervised CVAE
    :param labels: input labels for each image to be modified
    :param object_identifiers: identifiers for the object instances for each image
    :param percentage_kept: percentage of labels that will be kept
    :param canonical: whether we use a fixed split for the objects
    :return:
    """
    # Select the random seed if seed is chosen
    if canonical:
        np.random.seed(17)

    supervision = np.ones(len(labels), dtype=float)  # initialize the supervision
    unique_object_ids = np.unique(object_identifiers) # find object ids

    # Number of unlabelled data
    num_unlabelled_ids = int(np.round((1 - percentage_kept) * len(unique_object_ids)))

    # Select the portion of unique ids that will be selected as unlabelled
    identifiers_indexes = np.arange(len(unique_object_ids))
    np.random.shuffle(identifiers_indexes)
    erase_ids_index = identifiers_indexes[:num_unlabelled_ids]
    for num_id, erase_id in enumerate(erase_ids_index):
        remove_id = unique_object_ids[erase_id] # choose id of object to be removed
        # Identify indexes with the id of the object that will be removed
        removed_object_indexes = object_identifiers==remove_id
        # Fill the labels with arbitrary values and change the supervision
        labels[removed_object_indexes] = np.zeros(shape=labels[removed_object_indexes].shape)
        supervision[removed_object_indexes] = 0.0
    np.random.seed(None)
    return labels, supervision

def load_semisupervised_random_rotation(path, n_channels, percentage_kept=1.0, canonical=True, type_geometry ="angles", render_fold_number = 1, selection ="data"):

    assert percentage_kept <= 1.0 and percentage_kept >= 0.0, "percentage has to be between 0 and 1"
    assert type_geometry == "angles" or type_geometry == "quaternions", "not an existing load type {}".format(type)
    images_path = os.path.join(path, "images")
    identifiers_path = os.path.join(path, "identifiers")

    # Read image identifiers
    with open(identifiers_path + '/'+type_geometry+'.json') as json_file:
        identifiers = json.load(json_file)

    image_filenames = np.array(identifiers["model_identifier"])
    object_identifiers = [image_filename.split('_')[0] for image_filename in image_filenames]
    geometric_descriptor = np.array(identifiers[type_geometry])
    # Range for selecting files to be read by jumping render_fold_number
    arange_selection = np.arange(0, len(image_filenames), render_fold_number, dtype = int)
    # Select the files to be read
    image_filenames = image_filenames[arange_selection]
    geometric_descriptor = geometric_descriptor[arange_selection]





    # Define the supervision angles
    if type_geometry == "angles":
        labels = np.zeros((len(geometric_descriptor), 2))
        labels[:, 0] = np.cos(geometric_descriptor)
        labels[:, 1] = np.sin(geometric_descriptor)
    elif type_geometry == "quaternions":
        labels = np.array(geometric_descriptor)
    else:
        print("Type {} of geometry is not available".format(type_geometry))
        labels = None

    # Supervision
    if selection == "data":
        labels, supervision = select_supervision(labels, percentage_kept, canonical)
    elif selection == "object":
        labels, supervision = select_supervision_object_based(labels, object_identifiers, percentage_kept, canonical)

    path_list = [images_path + '/' + identifier.replace('.off', '.png') for identifier in image_filenames]

    images = np.array([imageio.imread(x)[:, :, :n_channels] for x in path_list])

    return images, labels, supervision

def load_semisupervised_rendered_images(resources_path, n_channels, percentage_kept, mode="render", canonical = True):
    if canonical:
        np.random.seed(17)
    images, labels = load_rendered_images(resources_path, n_channels, mode = mode)
    assert percentage_kept <= 1.0 and percentage_kept >= 0.0, "percentage has to be between 0 and 1"
    supervision = np.ones(len(labels), dtype=float)
    num_labelled_images = int(np.floor((1 - percentage_kept) * len(labels)))
    labels_indexes = np.arange(len(labels))
    np.random.shuffle(labels_indexes)
    erase_labels_index = labels_indexes[:num_labelled_images]
    labels[erase_labels_index,:] = np.zeros(shape=labels[erase_labels_index,:].shape)
    supervision[erase_labels_index] = 0.0
    np.random.seed(None)
    return images, labels, supervision

def separate_train_validation_data(supervision, train_percentage = 0.7, canonical = True):
    if canonical:
        np.random.seed(17)
    # Indexes for supervised/semisupervised data
    indexes_sup = (supervision == 1)
    indexes_unsup = (supervision == 0)
    amount_sup = int(np.sum(supervision))
    # Calculate the corresponding percentage of training data
    amount_train_data = int(np.round(train_percentage * amount_sup))

    # Define the tags for the supervised data between train 1 and val 0
    labels_sup = np.zeros(amount_sup)
    labels_sup[:amount_train_data] = np.ones(labels_sup[:amount_train_data].shape)
    np.random.shuffle(labels_sup) # shuffle the tags among supervised
    # Define the training/validation/unsupervised_labels
    train_val_labels = np.zeros(len(supervision))
    train_val_labels[indexes_sup] = labels_sup
    train_val_labels[indexes_unsup] = 2.0 # Unsupervised labels
    np.random.seed(None)
    return train_val_labels




def load_rendered_images_subset_renders_semisupervised(resources_path, n_channels, render_fold_number = 1, percentage_kept = 1.0, mode="render", canonical = True, selection = "data"):
    """
    Import images from the resources dir with certain number of channels
    :param resources_path: Dir path from were images are fetched
    :param n_channels: Number of colors for the images
    :return:
    """
    path_list = np.array(glob.glob(resources_path + '/*.png'))
    file_list = [os.path.basename(x) for x in path_list]
    object_identifiers = np.array([file.split("_")[1] for file in file_list])
    render_numbers = np.array([int(x.split("_")[-2]) for x in file_list])
    total_renders = np.amax(render_numbers)

    # Range for selecting files to be read by jumping render_fold_number
    arange_selection = np.arange(0,len(path_list), render_fold_number)

    # Select the files to be read
    path_list = path_list[arange_selection]
    render_numbers = render_numbers[arange_selection]
    object_identifiers = object_identifiers[arange_selection]
    # Read the data
    x_train = np.array([imageio.imread(x)[:, :, :n_channels] for x in path_list])
    # Create the labels
    if mode == "angles":
        labels = 2 * np.pi * render_numbers / total_renders
    elif mode == "circle":
        angles = 2 * np.pi * render_numbers / total_renders
        labels = np.zeros((len(angles), 2))
        labels[:, 0] = np.cos(angles)
        labels[:, 1] = np.sin(angles)
    else:
        labels = render_numbers

    # Select supervision either data-based (supervision is selected across data) or
    # object based(supervision is selected per object)
    if selection == "data":
        labels, supervision = select_supervision(labels, percentage_kept, canonical)
    elif selection == "object":
        labels, supervision = select_supervision_object_based(labels, object_identifiers, percentage_kept, canonical)
    else:
        labels, supervision = None, None
    return x_train, labels,  supervision


def load_semisupervised_rendered_images_train_val(resources_path, n_channels, percentage_kept, percentage_train, mode="render", canonical = True):
    images, labels, supervision = load_rendered_images(resources_path, n_channels, percentage_kept, mode = mode, canonical = canonical)

    assert percentage_kept <= 1.0 and percentage_kept >= 0.0, "percentage has to be between 0 and 1"
    supervision = np.ones(len(labels), dtype=float)
    num_labelled_images = int(np.round((1 - percentage_kept) * len(labels)))
    labels_indexes = np.arange(len(labels))
    np.random.shuffle(labels_indexes)
    erase_labels_index = labels_indexes[:num_labelled_images]
    labels[erase_labels_index,:] = np.zeros(shape=labels[erase_labels_index,:].shape)
    supervision[erase_labels_index] = 0.0
    np.random.seed(None)
    return images, labels, supervision





def load_rendered_images(resources_path, n_channels, mode="render"):
    """
    Import images from the resources dir with certain number of channels
    :param resources_path: Dir path from were images are fetched
    :param n_channels: Number of colors for the images
    :return:
    """
    path_list = list(glob.glob(resources_path + '/*.png'))
    file_list = [os.path.basename(x) for x in path_list]
    render_numbers = np.array([int(x.split("_")[-2]) for x in file_list])
    x_train = np.array([imageio.imread(x)[:, :, :n_channels] for x in path_list])
    if mode == "angles":
        labels = 2 * np.pi * render_numbers / np.amax(render_numbers)
    elif mode == "circle":
        angles = 2 * np.pi * render_numbers / np.amax(render_numbers)
        labels = np.zeros((len(angles), 2))
        labels[:, 0] = np.cos(angles)
        labels[:, 1] = np.sin(angles)
    else:
        labels = render_numbers
    return x_train, labels

def load_rendered_images_name(resources_path, n_channels, mode="render"):
    """
    Import images from the resources dir with certain number of channels
    :param resources_path: Dir path from were images are fetched
    :param n_channels: Number of colors for the images
    :return:
    """
    path_list = list(glob.glob(resources_path + '/*.png'))
    file_list = [os.path.basename(x) for x in path_list]
    object_identifier = [file.split("_")[1] for file in file_list]
    render_numbers = np.array([int(x.split("_")[-2]) for x in file_list])
    x_train = np.array([imageio.imread(x)[:, :, :n_channels] for x in path_list])
    if mode == "angles":
        labels = 2 * np.pi * render_numbers / np.amax(render_numbers)
    elif mode == "circle":
        angles = 2 * np.pi * render_numbers / np.amax(render_numbers)
        labels = np.zeros((len(angles), 2))
        labels[:, 0] = np.cos(angles)
        labels[:, 1] = np.sin(angles)
    else:
        labels = render_numbers
    return x_train, labels


def load_rendered_images_object_type(resources_path, n_channels, mode="render"):
    """
    Import images from the resources dir with certain number of channels
    :param resources_path: Dir path from were images are fetched
    :param n_channels: Number of colors for the images
    :return:
    """
    path_list = list(glob.glob(resources_path + '/*.png'))
    file_list = [os.path.basename(x) for x in path_list]
    object_list = []
    render_numbers = np.array([int(x.split("_")[-2]) for x in file_list])
    x_train = np.array([imageio.imread(x)[:, :, :n_channels] for x in path_list])
    if mode == "angles":
        labels = 2 * np.pi * render_numbers / np.amax(render_numbers)
    elif mode == "circle":
        angles = 2 * np.pi * render_numbers / np.amax(render_numbers)
        labels = np.zeros((len(angles), 2))
        labels[:, 0] = np.cos(angles)
        labels[:, 1] = np.sin(angles)
    else:
        labels = render_numbers
    return x_train, labels





def load_subset_rendered_images(resources_dir, n_channels, num_images):
    """
    Loads a num_images of the available renders in resources_dir
    :param resources_dir: directory where the renders can be found
    :param n_channels: number of color channels to be loaded in the image
    :param num_images: number of images to be loaded
    :return:
    """

    # Get the list of paths to images
    path_list = list(glob.glob(resources_dir + '/*.png'))
    file_list = [os.path.basename(x) for x in path_list]

    # Get the number of renders and order the data w.r.t. number of render
    render_numbers = np.array([int(x.split("_")[-2]) for x in file_list])
    path_list_ordered = np.array([path for _, path in sorted(zip(render_numbers, path_list))])
    render_numbers = np.array(sorted(render_numbers))

    # Take the subset of the path list and render numbers
    total_elements = len(path_list)
    step_size = total_elements // num_images
    assert step_size != 0, "Number of images requested is higher than the available images"
    sub_path_list = [path_list_ordered[i * step_size] for i in range(num_images)]
    sub_render_numbers = np.array([render_numbers[i * step_size] for i in range(num_images)])

    # Load the images
    x_train = np.array([imageio.imread(x)[:, :, :n_channels] for x in sub_path_list])
    return x_train, sub_render_numbers


def low_pass_filter(image, component_cutoff):
    """
    Do a low pass filter in Fourier domain with certain number of components cutoff after FFT(uses L1 norm in frequency
    space)
    :param image: Input image
    :param component_cutoff: Number of frequency (integer)
    :return:
    """
    filtered = np.copy(image)
    for color in range(image.shape[2]):
        image_fft = np.fft.fft2(image[:, :, color])
        shifted = np.fft.ifftshift(image_fft)
        for i in range(filtered.shape[0]):
            for j in range(filtered.shape[1]):
                if np.abs(i - image_fft.shape[0] / 2) + np.abs(j - image_fft.shape[1] / 2) > component_cutoff:
                    shifted[i, j] *= 0
        filtered[:, :, color] = np.fft.ifft2(np.fft.ifftshift(shifted)).real
    return filtered


def low_pass_filter_images(images, component_cutoff):
    """
    Low pass filter applied to a set of images with a certain number of components cutoff after FFT(uses L1 norm in
    frequency space)
    :param images:
    :param component_cutoff:
    :return:
    """
    filtered_images = [low_pass_filter(x, component_cutoff) for x in images]
    filtered_images = np.array(filtered_images)
    return filtered_images


def load_resize_save_image(image_path, x_resolution, y_resolution, target_dir, pil_extension):
    filename = os.path.basename(image_path)
    image = PIL.Image.open(image_path)
    image = image.resize((x_resolution, y_resolution), PIL.Image.ANTIALIAS)
    image.save(os.path.join(target_dir, filename), pil_extension)


def resize_images(load_path, target_dir, x_resolution, y_resolution, load_extension, save_extension):
    path_images = list(glob.glob(os.path.join(load_path, '*' + load_extension)))
    if save_extension == ".jpg":
        pil_extension = "JPEG"
    elif save_extension == ".png":
        pil_extension = "PNG"
    else:
        print("Save extension was not recognized {}".format(save_extension))
        return None
    [load_resize_save_image(path, x_resolution, y_resolution, target_dir, pil_extension) for path in path_images]
