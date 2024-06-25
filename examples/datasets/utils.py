import numpy as np
import numpy.typing as npt
from typing import Tuple, Any, List

def rotation_3x3_to_quaternion(R: npt.NDArray[Any]) -> Tuple[float, float, float, float]:
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return qw, qx, qy, qz

def get_sequence_on_sphere(n: int, radius: float) -> List[npt.NDArray[Any]]:
    '''Get a sequence of n points on the surface of a sphere of given radius.'''
    points = []

    phis = np.linspace(0, np.pi, n)
    thetas = np.linspace(0, 2 * np.pi, n)
    for phi, theta in zip(phis, thetas):
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)

        points.append(np.array([x, y, z]))

    return points

def sample_quaternion(n: int) -> npt.NDArray[Any]:
    #  h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))

    # Sample 3d angles from a uniform distribution
    # angles = np.random.rand(N, 3) * 2 * np.pi  # (N, 3) in (0, 2 * pi)
    # q = euler_to_quat(torch.from_numpy(angles))  # (N, 4) in (-1, 1)
    # q = q.to(torch.get_default_dtype())

    quat = np.random.randn(4, n)
    quat /= np.linalg.norm(quat, axis=0)
    return quat.T

def sample_3d_points(n: int, scene_size: float) -> npt.NDArray[Any]:
    '''Sample a set of 3D points uniformly within a scene.'''
    return np.random.uniform(-scene_size / 2, scene_size / 2, (n, 3))

def project_points_3d_to_2d(
        points_3d: npt.NDArray[Any],
        intrinsic_matrix: npt.NDArray[Any],
        extrinsic_matrix: npt.NDArray[Any]
        ):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    camera_coordinates = extrinsic_matrix @ points_3d_homogeneous.T
    points_2d_homogeneous = intrinsic_matrix @ camera_coordinates[:3, :]
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    return points_2d.T

