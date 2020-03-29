# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotationMatrixToEulerAngles2(R):
    """
    Method to calculate Euler angles based on the pdf written by Gregory G. Slabaugh
    In a certain situation two different angles are possible.
    """
    if np.abs(R[2,0])!=1:
        theta1 = -np.arcsin(R[2,0])
        theta2 = np.pi-theta1
        psi1 = np.arctan2(R[2,1]/np.cos(theta1), R[2,2]/np.cos(theta1))
        psi2 = np.arctan2(R[2,1]/np.cos(theta2), R[2,2]/np.cos(theta2))
        phi1 = np.arctan2(R[1,0]/np.cos(theta1), R[0,0]/np.cos(theta1))
        phi2 = np.arctan2(R[1,0]/np.cos(theta2), R[0,0]/np.cos(theta2))
    else:
        phi1 = 0
        phi2 = 0
        if R[2,0] == -1:
            theta1 = np.pi/2
            theta2 = np.pi/2
            psi1 = phi1+np.arctan2(R[0,1], R[0,2])
            psi2 = phi1+np.arctan2(R[0,1], R[0,2])
        else:
            theta1 = -np.pi/2
            theta2 = -np.pi/2
            psi1 = -phi1+np.arctan2(-R[0,1], -R[0,2])
            psi2 = -phi1+np.arctan2(-R[0,1], -R[0,2])
    return theta1,psi1, phi1, theta2,psi2,  phi2

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

def isorthogonal(matrix, tolerance = 1e-16):
    t_matrix = matrix.transpose()
    difference = np.sum(np.matmul(matrix,t_matrix)-np.eye(len(matrix)))**2
    return difference <= tolerance


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
    angles2 = np.zeros((num_samples, 6))
    for num_matrix, matrix in enumerate(orthogonal_matrices):
        # Make the determinant positive
        so3_matrices[num_matrix] = np.linalg.det(matrix) * matrix
        # Calculate corresponding angle with method1
        angles[num_matrix, :] = rotationMatrixToEulerAngles(so3_matrices[num_matrix])
        # Calculate corresponding angle with method2
        angles2[num_matrix, :] = rotationMatrixToEulerAngles2(so3_matrices[num_matrix])
    return so3_matrices, angles, angles2


def get_angles_from_matrices(matrices, algorithm=0):
    if algorithm == 0:
        angles = np.zeros((len(matrices), 3))
    elif algorithm == 1:
        angles = np.zeros((len(matrices), 6))
    else:
        angles = None
    for num_matrix, matrix in enumerate(matrices):
        if algorithm == 0:
            angles[num_matrix, :] = rotationMatrixToEulerAngles(matrix)
        if algorithm == 1:
            angles[num_matrix, :] = rotationMatrixToEulerAngles2(matrix)
    return angles


def angles_to_circular_colors(angles):
    colors = np.zeros((len(angles), 3))
    colors[:, 0] = 0.5 + np.cos(angles) / 2
    colors[:, 1] = 0.5 + np.sin(angles) / 2
    return colors


def plot_angles(angles, color_angle_number):
    colors = angles_to_circular_colors(angles[:, color_angle_number])
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter(angles[:, 0], angles[:, 1], angles[:, 2], c=colors)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\varphi$")
    ax.set_zlabel(r"$\psi$")


def plot_angles_from_matrices(matrices, algorithm, color_angle_number):
    angles = get_angles_from_matrices(matrices, algorithm)
    colors = angles_to_circular_colors(angles[:, color_angle_number])
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter(angles[:, 0], angles[:, 1], angles[:, 2], c=colors)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\varphi$")
    ax.set_zlabel(r"$\psi$")


####### RP3 Plotting #########
def get_angles(matrices):
    angles = np.array([np.arccos((np.clip(np.trace(matrix), a_min=-1.0, a_max=3.0) - 1) / 2.0) for matrix in matrices])
    return angles


def get_axis(matrices):
    axes = np.zeros((len(matrices), 3))
    for num_matrix, matrix in enumerate(matrices):
        difference = matrix - matrix.transpose()
        axes[num_matrix, 0] = difference[2, 1]
        axes[num_matrix, 1] = difference[0, 2]
        axes[num_matrix, 2] = difference[1, 0]
    axes = axes / (2 * np.expand_dims(np.sin(get_angles(matrices)), 1))
    return axes


def get_rp3(matrices):
    angles = get_angles(matrices)
    axes = np.zeros((len(matrices), 3))
    quotient_sine = np.zeros((len(matrices), 1))
    #non_orthogonal = 0
    for num_matrix, matrix in enumerate(matrices):
        #if not(isorthogonal(matrix, tolerance = 1e-10)):
        #    non_orthogonal += 1
        difference = matrix - matrix.transpose()
        if np.isclose(angles[num_matrix], 0.0):
            quotient_sine[num_matrix] = 0.0
            axes[num_matrix, :] = np.zeros(3)
        elif np.abs(angles[num_matrix]- np.pi)<= 0.0133:
            quotient_sine[num_matrix] = np.pi
            B = (matrix + np.eye(3)) / 2.0
            axes[num_matrix, 0] = np.sqrt(B[0, 0])
            # axes[num_matrix, 0] = np.sign(-B[0,2])
            axes[num_matrix, 1] = np.sqrt(B[1, 1])
            axes[num_matrix, 1] *= np.sign(B[0, 1]) * np.sign(axes[num_matrix, 0])
            axes[num_matrix, 2] = np.sqrt(B[2, 2])
            axes[num_matrix, 2] *= np.sign(axes[num_matrix, 1]) * np.sign(B[1, 2])
        else:
            quotient_sine[num_matrix] = angles[num_matrix] / (2 * np.sin(angles[num_matrix]))
            axes[num_matrix, 0] = difference[2, 1]
            axes[num_matrix, 1] = difference[0, 2]
            axes[num_matrix, 2] = difference[1, 0]
    axes = quotient_sine * axes / np.pi
    #print("Number of non orthogonal matrices {}".format(non_orthogonal))
    return axes


def plot_rp3(matrices, angles):
    rp3_vectors = get_rp3(matrices)
    fig = plt.figure(figsize=(5,5))
    ax = Axes3D(fig)
    ax.scatter(rp3_vectors[:, 0], rp3_vectors[:, 1], rp3_vectors[:, 2], c=angles)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    # Plot a sphere
    samples = 100
    theta = 2 * np.pi * np.linspace(0, 1, samples)
    phi = np.pi * np.linspace(0, 1, samples)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.07, linewidth=0, shade=True)


### Get matrices from axis and angle

def axis_angle_to_matrix(axes, angles):
    basis_matrices = np.zeros((3, 3, 3))
    basis_matrices[0, 1, 2] = -1
    basis_matrices[0, 2, 1] = 1
    basis_matrices[1, 0, 2] = 1
    basis_matrices[1, 2, 0] = -1
    basis_matrices[2, 0, 1] = -1
    basis_matrices[2, 1, 0] = 1
    basis_matrices = np.expand_dims(basis_matrices, axis=0)
    axes = np.expand_dims(axes, axis=-1)
    axes = np.expand_dims(axes, axis=-1)
    angles = np.expand_dims(angles, axis=-1)
    matrix_combination = axes * basis_matrices
    matrix_combination = np.sum(matrix_combination, axis=1)
    matrix_combination_square = np.copy(matrix_combination)
    for num_matrix, matrix in enumerate(matrix_combination):
        matrix_combination_square[num_matrix] = np.matmul(matrix, matrix)
    identity_matrix = np.expand_dims(np.eye(3), axis=0)
    rotation_matrix = identity_matrix + np.sin(angles) * matrix_combination + (
                1 - np.cos(angles)) * matrix_combination_square
    return basis_matrices, axes, angles, matrix_combination, matrix_combination_square, rotation_matrix

def rotate_matrices(matrices, rotation):
    rotated = [np.matmul(rotation, matrix) for matrix in matrices]
    return rotated

def so3_projection(matrix):
    u, s, vh = np.linalg.svd(matrix)
    # Orthogonal matrix
    orthogonal_matrix = np.matmul(u, vh)
    so3_matrix = np.linalg.det(orthogonal_matrix) * orthogonal_matrix
    return so3_matrix
def random_close(matrices, scale):
    random_close_matrices = []
    for num_matrix, matrix in enumerate(matrices):
        random_close_matrices.append(so3_projection(matrix+np.random.normal(loc = 0.0, scale = scale, size = (3,3))))
    random_close_matrices = np.array(random_close_matrices)
    return random_close_matrices
