import types
import numpy as np

SMALL_GRID = np.array([[0.5581934, -0.82923452, 0.02811105],
                       [0.85314824, -0.15111695, 0.49930127],
                       [0.33363164, 0.92444637, 0.18463162],
                       [-0.79055406, -0.61207313, -0.0197677],
                       [0.54115503, -0.6990933, -0.46735403],
                       [0.2572988, 0.76735657, 0.587334],
                       [-0.71900497, 0.13915788, -0.68093094],
                       [-0.01575003, -0.98767151, -0.15574634],
                       [0.11254023, 0.38710762, 0.91514064],
                       [0.65727848, -0.25558372, 0.70898658]])

GRID = np.array([[-0.3889949, 0.25055803, -0.88651207],
                 [-0.32215506, -0.01322882, -0.94659448],
                 [0.02367464, 0.33125212, 0.94324522],
                 [0.14352843, 0.98208741, -0.12208155],
                 [-0.33731424, -0.02639622, 0.94102197],
                 [-0.71371608, -0.00913657, 0.70037553],
                 [0.27543912, -0.56977281, 0.77426884],
                 [0.9494808, -0.27132451, 0.15769981],
                 [0.80534178, 0.58419015, -0.10072976],
                 [-0.71157706, 0.56389324, -0.41914496],
                 [-0.63249914, 0.26813677, -0.72666878],
                 [0.03723451, 0.50673582, 0.86129693],
                 [0.58980619, -0.66773587, 0.45415578],
                 [0.91747066, -0.04932981, 0.39473302],
                 [-0.63278129, 0.50029337, -0.59101132],
                 [0.71905015, 0.63130778, 0.29054667],
                 [0.41343092, 0.4093163, 0.81334804],
                 [0.69237513, 0.53825332, 0.48052059],
                 [-0.66960206, 0.47597747, -0.57015659],
                 [0.99181207, -0.11223239, -0.06093197],
                 [-0.97539005, -0.15734202, -0.15445952],
                 [-0.73199196, -0.13932144, -0.66691627],
                 [0.09859593, 0.79242886, 0.60194298],
                 [0.86632913, 0.4181876, -0.27311714],
                 [0.1962053, -0.62076618, 0.75904732],
                 [-0.52996612, 0.68788398, 0.49593502],
                 [-0.57038469, -0.72583872, 0.38447297],
                 [0.94680602, 0.32009305, 0.03314802],
                 [0.2062331, -0.91981748, -0.33377194],
                 [0.00492786, -0.90938187, -0.41593308],
                 [-0.72349802, 0.606303, -0.33007164],
                 [-0.64359981, -0.50221586, -0.57754525],
                 [0.21802296, -0.89754906, 0.38323841],
                 [0.73938226, -0.37591704, -0.55856983],
                 [-0.45225152, -0.07171458, -0.88900258],
                 [0.52498364, -0.47577634, -0.70571174],
                 [0.90728605, 0.10420356, 0.40739863],
                 [0.08465876, 0.98327685, -0.16124373],
                 [-0.51854311, 0.63194738, -0.57598225],
                 [0.60001613, 0.5703173, 0.56099806],
                 [-0.25533381, -0.37958125, -0.88922591],
                 [-0.41425908, 0.37349873, 0.82999284],
                 [-0.09570411, 0.76619074, -0.63544667],
                 [-0.56434898, -0.42910009, -0.7052541],
                 [-0.65264073, -0.75588422, 0.05195301],
                 [0.00409419, -0.82815987, -0.56047699],
                 [0.39340692, -0.35219701, 0.84922804],
                 [-0.40230759, -0.71564088, 0.57096999],
                 [-0.10278411, 0.48295417, -0.86959226],
                 [0.54247975, 0.83325265, 0.10679769],
                 [0.92379565, -0.32112374, -0.20852131],
                 [0.15224038, 0.21710568, -0.96420329],
                 [-0.10514164, 0.79083545, -0.60292995],
                 [-0.21746656, -0.37151197, 0.90260021],
                 [0.3109654, 0.46004438, -0.8316608],
                 [-0.23916412, -0.49089814, -0.83774671],
                 [0.29699089, -0.89150092, -0.34208554],
                 [0.14917159, -0.21317452, -0.96555914],
                 [0.22686163, 0.10414401, -0.96834283],
                 [0.7175495, 0.3904845, -0.57675347],
                 [-0.13066132, -0.8137806, 0.56629387],
                 [-0.71179249, -0.32746321, 0.62138498],
                 [0.2548561, -0.6620188, 0.70482585],
                 [-0.60030469, 0.75526266, 0.26308287],
                 [-0.95210526, 0.22242061, 0.2098205],
                 [0.63696893, -0.7544887, -0.15816883],
                 [0.80888482, -0.48146657, -0.33748375],
                 [-0.22148124, 0.84744604, 0.48247411],
                 [0.03338003, 0.57086839, -0.82036276],
                 [0.35481394, 0.93054951, 0.09046918],
                 [-0.57813618, 0.69862557, 0.42152208],
                 [0.39088467, 0.77462782, 0.49715281],
                 [0.81270012, 0.58214702, -0.02496695],
                 [0.30466405, 0.34525589, -0.88768135],
                 [-0.08086346, 0.76866636, 0.63451803],
                 [0.79030596, -0.60912802, 0.06617809],
                 [0.40744375, -0.69386156, -0.59375561],
                 [-0.93496061, 0.30292708, 0.18461811],
                 [-0.99092609, -0.04953639, -0.12494651],
                 [0.61112374, 0.7797983, 0.13580276],
                 [0.26064656, -0.28859611, 0.92129021],
                 [-0.5490118, -0.65302497, -0.52167464],
                 [-0.842748, -0.50960614, -0.17342834],
                 [0.6244172, 0.55517995, 0.5494346],
                 [-0.06157987, 0.95344137, 0.29522445],
                 [0.63583035, 0.57326159, -0.51680839],
                 [0.56591439, 0.0229997, 0.82414314],
                 [0.71834931, 0.29183486, -0.63151142],
                 [0.47572203, -0.35993717, -0.80257946],
                 [-0.29635979, -0.4446486, 0.84525647],
                 [0.66083764, 0.74029486, 0.12351978],
                 [0.45341129, -0.85099596, 0.26499828],
                 [-0.64599992, -0.30902696, 0.69798742],
                 [-0.57768232, 0.64783958, -0.49657529],
                 [-0.64443451, -0.6510948, 0.40097347],
                 [0.94840987, 0.05513716, -0.31221566],
                 [-0.64239476, 0.62933839, -0.4373353],
                 [0.35569241, 0.86516397, -0.35351691],
                 [-0.47306257, 0.28775386, -0.83271215],
                 [-0.3091001, 0.52021522, 0.79613645]])


def sample(batch_size, f, grid=None):
    if grid is None:
        grid = SMALL_GRID.copy()

    repr_vectors = np.random.normal(size=(batch_size, 4))
    axes, angles = repr_4d_to_axis_angle(repr_vectors)
    return f(multiply_from_left(repr_vectors, grid, invert=True)), repr_vectors, axes, angles


def repr_4d_to_axis_angle(vectors):
    length = vectors.shape[0]
    vectors = vectors / np.reshape(np.linalg.norm(vectors, axis=-1), (length, 1))

    angles = np.arccos(vectors[:, 0])

    vectors_3d = vectors[:, 1:4]
    vectors_3d = vectors_3d / np.reshape(np.linalg.norm(vectors_3d, axis=-1), (length, 1))

    return vectors_3d, angles


def spherical_axis_angle_to_repr_4d(phis, thetas, angles):
    length = phis.shape[0]

    repr_4d = np.zeros(shape=(length, 4))

    repr_4d[:, 0] = np.cos(angles)
    repr_4d[:, 1] = np.cos(phis) * np.sin(thetas) * np.sin(angles)
    repr_4d[:, 2] = np.sin(phis) * np.sin(thetas) * np.sin(angles)
    repr_4d[:, 3] = np.cos(thetas) * np.sin(angles)

    return repr_4d


def vector_to_rotation(vectors, invert=False):
    """Produces a rotation matrix from a vector in R^4
    
    We use the axis-angle representation."""

    # First normalize the vectors
    length = vectors.shape[0]
    vectors = vectors / np.reshape(np.linalg.norm(vectors, axis=-1), (length, 1))

    if invert:
        angles = - np.arccos(vectors[:, 0])
    else:
        angles = np.arccos(vectors[:, 0])

    vectors_3d = vectors[:, 1:4]
    vectors_3d = vectors_3d / np.reshape(np.linalg.norm(vectors_3d, axis=-1), (length, 1))

    return axis_angle(vectors_3d, angles)


def cross_product_matrices(vectors):
    """Return corresponding cross product matrices for an array of vectors 
    """
    length = vectors.shape[0]
    result = np.tensordot(np.cross(vectors, unit_vectors(length, 3, 0)), np.array([1., 0., 0.]), axes=0) \
             + np.tensordot(np.cross(vectors, unit_vectors(length, 3, 1)), np.array([0., 1., 0.]), axes=0) \
             + np.tensordot(np.cross(vectors, unit_vectors(length, 3, 2)), np.array([0., 0., 1.]), axes=0)
    return result


def multiply_from_left(repr_vectors, vectors, invert=False):
    rot_matrices = np.transpose(vector_to_rotation(repr_vectors, invert), axes=(0, 2, 1))
    pre_result = np.tensordot(rot_matrices, vectors, axes=((-2,), (-1,)))
    result = np.transpose(pre_result, axes=(0, 2, 1))
    return result


def axis_angle(vectors, angles):
    """Return rotation matrices according to axis-angle representation
    """
    length = vectors.shape[0]
    matrices = cross_product_matrices(vectors)

    identities = np.tile(np.identity(3), (length, 1, 1))

    rotation_matrices = identities \
                        + np.reshape(np.sin(angles), (length, 1, 1)) * matrices \
                        + np.reshape(1 - np.cos(angles), (length, 1, 1)) * np.matmul(matrices, matrices)
    return rotation_matrices


def unit_vectors(n_rows, n_cols, i):
    """Return n_rows standard unit vectors of n_cols component, st ith component is 1
    
    >>> eps = 0.0001
    
    >>> np.linalg.norm(unit_vectors(2,3,1) - np.array([[0,1,0],[0,1,0]]) ) < eps 
    True
    """
    result = np.zeros((n_rows, n_cols))
    result[:, i] = 1
    return result


def stereographic_projection(z_values):
    z_upper = np.reshape(-np.sign(z_values[:, 0]), (-1, 1)) * z_values
    dims = z_values.shape[1]
    z_0 = z_upper[:, 0]
    stereo_proj = np.copy(z_upper[:, 1:dims]) / (1 - z_0[:, np.newaxis])
    return stereo_proj


def example_function(values):
    return values[..., 0] * values[..., 0] * values[..., 0] + 0.3 * values[..., 0] + values[..., 2] * values[..., 2] * \
           values[..., 2]


def ex_function_deg_2(values):
    return values[..., 0] * values[..., 1] + 0.8 * values[..., 2] + 0.5 * values[..., 0] * values[..., 0]


class random_polynomial:

    def __init__(self):
        self.degree = 3
        gamma = [1, 1, 0.8, 0.3]
        self._coeff = np.zeros(shape=(self.degree + 1, self.degree + 1, self.degree + 1))
        self._coeff[0, 0, 1] = gamma[1] * np.random.normal(1)  # z
        self._coeff[0, 0, 2] = gamma[2] * np.random.normal(1)  # z^2
        self._coeff[0, 0, 3] = gamma[3] * np.random.normal(1)  # z^3
        self._coeff[0, 1, 0] = gamma[1] * np.random.normal(1)  # mu_z
        self._coeff[0, 1, 1] = gamma[2] * np.random.normal(1)  # yz
        self._coeff[0, 1, 2] = gamma[3] * np.random.normal(1)  # yz^2
        self._coeff[0, 2, 0] = gamma[2] * np.random.normal(1)  # mu_z^2
        self._coeff[0, 2, 1] = gamma[3] * np.random.normal(1)  # mu_z^2 z
        self._coeff[0, 3, 0] = gamma[3] * np.random.normal(1)  # mu_z^3
        self._coeff[1, 0, 0] = gamma[1] * np.random.normal(1)  # x
        self._coeff[1, 0, 1] = gamma[2] * np.random.normal(1)  # xz
        self._coeff[1, 0, 2] = gamma[3] * np.random.normal(1)  # xz^2
        self._coeff[1, 1, 0] = gamma[2] * np.random.normal(1)  # xy
        self._coeff[1, 1, 1] = gamma[3] * np.random.normal(1)  # xyz
        self._coeff[1, 2, 0] = gamma[3] * np.random.normal(1)  # xy^2
        self._coeff[2, 0, 0] = gamma[2] * np.random.normal(1)  # x^2
        self._coeff[2, 0, 1] = gamma[3] * np.random.normal(1)  # x^2z
        self._coeff[2, 1, 0] = gamma[3] * np.random.normal(1)  # x^2y
        self._coeff[3, 0, 0] = gamma[3] * np.random.normal(1)  # x^3

    def __call__(self, values):
        x = values[..., 0]
        y = values[..., 1]
        z = values[..., 2]
        return self._coeff[0, 0, 1] * z + \
               self._coeff[0, 0, 2] * z * z + \
               self._coeff[0, 0, 3] * z * z * z + \
               self._coeff[0, 1, 0] * y + \
               self._coeff[0, 1, 1] * y * z + \
               self._coeff[0, 1, 2] * y * z * z + \
               self._coeff[0, 2, 0] * y * y + \
               self._coeff[0, 2, 1] * y * y * z + \
               self._coeff[0, 3, 0] * y * y * y + \
               self._coeff[1, 0, 0] * x + \
               self._coeff[1, 0, 1] * x * z + \
               self._coeff[1, 0, 2] * x * z * z + \
               self._coeff[1, 1, 0] * x * y + \
               self._coeff[1, 1, 1] * x * y * z + \
               self._coeff[1, 2, 0] * x * y * y + \
               self._coeff[2, 0, 0] * x * x + \
               self._coeff[2, 0, 1] * x * x * z + \
               self._coeff[2, 1, 0] * x * x * y + \
               self._coeff[3, 0, 0] * x * x * x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
