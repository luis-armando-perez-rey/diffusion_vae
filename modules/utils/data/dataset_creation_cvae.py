# System imports
import os

# Standard imports
import numpy as np
import itertools
import glob
import imageio
import json

def select_supervision(labels, percentage_kept, seed=17):
    np.random.seed(seed)
    # Supervision
    supervision = np.ones(len(labels), dtype=float)
    num_labelled_images = int(np.round((1 - percentage_kept) * len(labels)))
    labels_indexes = np.arange(len(labels))
    np.random.shuffle(labels_indexes)
    erase_labels_index = labels_indexes[:num_labelled_images]
    false_label = np.ones(shape=labels[erase_labels_index, :].shape)
    false_label = false_label / np.expand_dims(np.linalg.norm(false_label, axis=-1), axis=-1)
    labels[erase_labels_index, :] = false_label
    supervision[erase_labels_index] = 0.0
    np.random.seed(None)
    return labels, supervision

def select_by_identifier(identifiers, percentage_labeled, seed = 17):
    np.random.seed(seed)
    unique_identifiers = np.unique(identifiers)
    np.random.shuffle(unique_identifiers)

    # Separate the labeled and the unlabeled identifiers
    num_labeled_identifiers = int(np.round(percentage_labeled * len(unique_identifiers)))
    labeled_identifiers = unique_identifiers[:num_labeled_identifiers]
    unlabeled_identifiers = unique_identifiers[num_labeled_identifiers:]
    print("Number of labeled identifiers {}".format(len(labeled_identifiers)))
    print("Number of unlabeled identifiers {}".format(len(unlabeled_identifiers)))
    np.random.seed(None)
    return labeled_identifiers, unlabeled_identifiers

def create_boolean_identifiers(identifiers, target_identifiers):
    """
    Creates a boolean with True values in the places where the identifiers belong to the
    target identifiers array
    :param identifiers: identifiers array that identifies each of the target values
    :param target_identifiers: set of target identifiers to be selected
    :return:
    """
    boolean = np.zeros(len(identifiers), dtype=bool)
    for identifier in target_identifiers:
        boolean += (identifiers == identifier)
    return boolean


def select_supervision_object_based(labels, object_identifiers, percentage_kept, seed=17):
    """
    This function produces the labels and the supervision array for training the semi-supervised CVAE
    :param labels: input labels for each image to be modified
    :param object_identifiers: identifiers for the object instances for each image
    :param percentage_kept: percentage of labels that will be kept
    :param seed: whether we use a fixed split for the objects
    :return:
    """
    # Select the random seed if seed is chosen
    np.random.seed(seed)

    supervision = np.ones(len(labels), dtype=float)  # initialize the supervision
    unique_object_ids = np.unique(object_identifiers) # find object ids

    # Number of unlabelled data
    num_unlabelled_ids = int(np.round((1 - percentage_kept) * len(unique_object_ids)))

    # Select the portion of unique ids that will be marked as unlabelled
    identifiers_indexes = np.arange(len(unique_object_ids))
    np.random.shuffle(identifiers_indexes)
    erase_ids_index = identifiers_indexes[:num_unlabelled_ids]
    for num_id, erase_id in enumerate(erase_ids_index):
        remove_id = unique_object_ids[erase_id] # choose id of object to be removed
        # Identify indexes with the id of the object that will be removed
        removed_object_indexes = object_identifiers == remove_id
        # Fill the labels with arbitrary values and change the supervision
        false_label = np.zeros(shape=labels[removed_object_indexes].shape)
        false_label[:,-1] = 1
        labels[removed_object_indexes] = false_label
        supervision[removed_object_indexes] = 0.0
    np.random.seed(None)
    return labels, supervision

def create_labels(angles, mode):
    if mode == "angles":
        labels = angles
    elif mode == "circle":
        labels = np.zeros((len(angles), 2))
        labels[:, 0] = np.cos(angles)
        labels[:, 1] = np.sin(angles)
    elif mode == "quaternion":
        labels = angles
    else:
        labels = angles
    return labels

def create_supervision(labels, object_identifiers, percentage_kept, selection, canonical):
    if selection == "data":
        labels, supervision = select_supervision(labels, percentage_kept, canonical)
    elif selection == "object":
        labels, supervision = select_supervision_object_based(labels, object_identifiers, percentage_kept, canonical)
    else:
        labels, supervision = None, None
    return labels, supervision

def load_identifiers(identifiers_path):
    """
    Load the identifiers that describe each of the rendered images from ModelNet40.
    The identifiers folder is saved as a .json file
    :param identifiers_path: path where the .json file with identifiers is stored
    :return: object_types (class of object),
    model_identifiers (instance ID),
    render_numbers (render number value),
    image_filenames
    """
    with open(os.path.join(identifiers_path ,'identifiers.json')) as json_file:
        identifiers = json.load(json_file)
    render_numbers = np.array(identifiers["render_number"])
    model_identifiers = np.array(identifiers["model_identifier"])
    object_types = np.array(identifiers["object_type"])
    image_filenames = np.array(identifiers["filename"])
    try:
        labels = np.array(identifiers["labels"])
    except:
        labels = None
    return object_types, model_identifiers, labels, render_numbers, image_filenames


def create_filenames_combinations(unique_object_types, unique_object_identifiers, unique_selected_renders):
    combinations = np.array(list(itertools.product(unique_object_types, unique_object_identifiers, unique_selected_renders)))
    filenames = [combination[0]+'_'+combination[1]+'_'+str(combination[2])+'_.png' for combination in combinations]
    return filenames, combinations

def load_semisupervised(resources_path, n_channels, render_fold_number=1, percentage_kept=1.0, mode="circle", seed=17,
                        selection="object"):
    """
    Import images from the resources dir with certain number of channels
    :param resources_path: Dir path from were images are fetched
    :param n_channels: Number of colors for the images
    :return:
    """
    images_path = os.path.join(resources_path, "images")
    identifiers_path = os.path.join(resources_path, "identifiers")

    # Read image identifiers
    _, identifiers, _, render_numbers, _ = load_identifiers(identifiers_path)

    # Identify unique identifiers
    unique_identifiers = np.unique(identifiers)
    unique_render_numbers = np.unique(render_numbers)
    total_renders = np.amax(unique_render_numbers)
    unique_selected_render_numbers = np.arange(0, total_renders+1, render_fold_number, dtype=int)

    # Load the images, render numbers and labels
    selected_images, selected_render_numbers, selected_labels, selected_identifiers = load_from_identifiers_renders(resources_path, unique_identifiers, unique_selected_render_numbers, n_channels)



    print(len(unique_render_numbers))
    print(len(np.unique(selected_render_numbers)))

    assert len(unique_render_numbers)// render_fold_number == len(
        np.unique(selected_render_numbers)) , "Number of renders is not folded"

    # Define the labels for supervision

    labels = create_labels(selected_labels, mode)

    # Define the supervision tags
    labels, supervision = create_supervision(labels, selected_identifiers, percentage_kept, selection, seed)


    return selected_images, labels, supervision

def get_labeled_unlabeled_identifiers(resources_path, percentage_labeled=1.0, seed=17):
    """
    Produce two arrays of identifiers: one corresponding to the labeled data and one corresponding to the
    unlabeled data. The identifiers are loaded from a given resources path
    :param resources_path: Dir path from were images are fetched
    :param percentage_labeled: percentage of labeled data to be used
    :return:
    """
    np.random.seed(seed)
    identifiers_path = os.path.join(resources_path, "identifiers")

    # Read image identifiers

    _, identifiers,_ ,_, _ = load_identifiers(identifiers_path)

    # Identify unique identifiers
    unique_identifiers = np.unique(identifiers)

    # Count amount of unlabelled identifiers
    num_unlabelled_ids = int(np.round((1 - percentage_labeled) * len(unique_identifiers)))

    # Create labeled, unlabeled indexes to retrieve identifiers
    identifiers_indexes = np.arange(len(unique_identifiers))
    np.random.shuffle(identifiers_indexes)
    unlabeled_indexes = identifiers_indexes[:num_unlabelled_ids]
    labeled_indexes = identifiers_indexes[num_unlabelled_ids:]

    # Select corresponding identifiers
    unlabeled_identifiers = unique_identifiers[unlabeled_indexes]
    labeled_identifiers = unique_identifiers[labeled_indexes]

    # Restore random seed
    np.random.seed(None)
    return labeled_identifiers, unlabeled_identifiers



def load_from_identifiers_renders(path, target_identifiers, target_renders, n_channels):
    """
    Load the images, render numbers and labels (angles) from a certain path with respect to the target identifiers
    (target identifiers are the specific instance ids) and with respect to the target renders.
    :param path: path where the data is stored
    :param target_identifiers: target identifiers for loading
    :param target_renders: target renders
    :param n_channels:
    :return:
    """
    # Define the corresponding paths for images and identifiers
    identifiers_path = os.path.join(path, "identifiers")
    images_path = os.path.join(path, "images")

    # Load the data identifiers
    _, model_identifiers, labels, render_numbers,image_filenames = load_identifiers(identifiers_path)

    # Create boolean arrays for selecting w.r.t. identifiers and render numbers
    bool_selection_models = np.zeros(len(model_identifiers), dtype=bool)
    bool_selection_renders = np.zeros(len(model_identifiers), dtype=bool)
    for identifier in target_identifiers:
        bool_selection_models += model_identifiers == identifier

    for render in target_renders:
        bool_selection_renders += render_numbers == render

    # Perform AND operation between selected models and renders
    bool_selection = bool_selection_models * bool_selection_renders

    # Select the renders and images
    selected_render_numbers = render_numbers[bool_selection]
    selected_image_filenames = image_filenames[bool_selection]
    selected_labels = labels[bool_selection]
    selected_identifiers = model_identifiers[bool_selection]
    selected_images = np.array([imageio.imread(os.path.join(images_path, filename))[:, :, :n_channels] for filename in
                                selected_image_filenames])

    return selected_images, selected_render_numbers, selected_labels, selected_identifiers



def join_images_labels_supervision(images1, images2, labels1, labels2, supervision1, supervision2):
    """
    Join the images, labels and supervision obtained from other methods. Joining is done on the batch dimension (first).
    :param images1: First array of images
    :param images2: Second array of images
    :param labels1: First array of labels
    :param labels2: Second array of labels
    :param supervision1: First array of supervision labels
    :param supervision2: Second array of supervision labels
    :return:
    """
    images = np.concatenate((images1, images2), axis = 0)
    labels = np.concatenate((labels1, labels2), axis=0)
    supervision = np.concatenate((supervision1, supervision2), axis=0)
    return images, labels, supervision













def load_semisupervised_random_rotation(path, n_channels, percentage_kept=1.0, seed=17, type_geometry ="angles", render_fold_number = 1, selection ="data"):

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
        labels, supervision = select_supervision(labels, percentage_kept, seed)
    elif selection == "object":
        labels, supervision = select_supervision_object_based(labels, object_identifiers, percentage_kept, seed)

    path_list = [images_path + '/' + identifier.replace('.off', '.png') for identifier in image_filenames]

    images = np.array([imageio.imread(x)[:, :, :n_channels] for x in path_list])

    return images, labels, supervision

def load_semisupervised_rendered_images(resources_path, n_channels, percentage_kept, mode="render", seed = 17):

    np.random.seed(seed)
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

def get_train_val_indicators(supervision, train_percentage = 0.7, seed = 17):
    """
    Creates an array with values 0 (validation) 1 (training) 2 (unlabelled) data.
    :param supervision: supervision array with values 0 (unlabelled) and 1 (labelled) for each training value
    :param train_percentage: percentage between 0 and 1 for number of training data to be selected
    :param seed: seed value for random. If None, then truly random.
    :return:
    """
    # Select the random seed.
    np.random.seed(seed)
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
    train_val_labels = np.zeros(len(supervision)) # validation data
    train_val_labels[indexes_sup] = labels_sup # train data
    train_val_labels[indexes_unsup] = 2.0 # unsupervised data
    np.random.seed(None)
    return train_val_labels


def separate_train_validation(x_train, labels, supervision, train_percentage):
    # Separate train validation
    train_val_indicators = get_train_val_indicators(supervision, train_percentage)
    # Indexes
    train_split_index = (train_val_indicators == 1) + (train_val_indicators == 2)
    val_split_index = (train_val_indicators == 0)
    # Data
    x_train_split = x_train[train_split_index]
    x_val_split = x_train[val_split_index]
    # Labels
    labels_train_split = labels[train_split_index]
    labels_val_split = labels[val_split_index]
    # Supervision
    supervision_train_split = supervision[train_split_index]
    supervision_val_split = supervision[val_split_index]
    train_data = (x_train_split, labels_train_split, supervision_train_split)
    validation_data = ([x_val_split, labels_val_split, supervision_val_split], [x_val_split, labels_val_split])
    return train_data, validation_data


def load_rendered_images_subset_renders_semisupervised(resources_path, n_channels, render_fold_number = 1, percentage_kept = 1.0, mode="render", seed = 17, selection ="data"):
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
        labels, supervision = select_supervision(labels, percentage_kept, seed)
    elif selection == "object":
        labels, supervision = select_supervision_object_based(labels, object_identifiers, percentage_kept, seed)
    else:
        labels, supervision = None, None
    return x_train, labels,  supervision


def load_semisupervised_rendered_images_train_val(resources_path, n_channels, percentage_kept, percentage_train, mode="render", seed = 17):
    images, labels, supervision = load_rendered_images(resources_path, n_channels, percentage_kept, mode = mode, seed = seed)

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

