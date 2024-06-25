import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from datasets.utils import (
        get_sequence_on_sphere,
        sample_3d_points,
        project_points_3d_to_2d,
        )


class SyntheticParser:
    """Synthetic parser."""

    def __init__(self, cfg, normalize: bool = False):
        self.cfg = cfg

        self.data_dir = Path(cfg.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.scene_scale = 1.0
        self.normalize = normalize
        self.test_every = cfg.test_every

        self.image_paths = [self.data_dir / f"{i:08d}.png" for i in range(cfg.num_imgs)]

        # Generate extrinsics
        self.cam2worlds = self._get_cam2worlds()

        # Generate intrinsics
        # For now all cameras have the same intrinsics
        self.camera_ids = np.zeros(len(self.cam2worlds), dtype=np.int32)
        self.Ks_dict = self._get_Ks_dict()
        self.imsize_dict = {0: (cfg.im_res, cfg.im_res)}

        # Generate points
        self.points = self._generate_3d_points()  
        self.points_rgb = np.array(self.cfg.points_rgb, dtype=np.uint8)

        # Create images
        self.images = self._create_images()

    def _get_cam2worlds(self):
        camera_positions = get_sequence_on_sphere(self.cfg.num_imgs, self.cfg.radius)

        camera_target = np.array([0, 0, 0])
        camera_up = np.array([0, 1, 0])

        cam2worlds = []

        for cam_pos in camera_positions:
            forward = (camera_target - cam_pos) / np.linalg.norm(camera_target - cam_pos)
            right = np.cross(camera_up, forward) / np.linalg.norm(np.cross(camera_up, forward))
            up = np.cross(forward, right)

            cam2world = np.eye(4)
            cam2world[:3, :3] = np.vstack([right, up, forward])
            cam2world[:3, 3] = -cam2world[:3, :3] @ cam_pos

            cam2worlds.append(cam2world)

        cam2worlds = np.stack(cam2worlds, axis=0)

        return cam2worlds

    def _get_Ks_dict(self):
        FoVx = 30.0 / 360 * 2 * np.pi
        FoVy = 30.0 / 360 * 2 * np.pi

        fx = 0.5 / np.tan(FoVx / 2) * self.cfg.im_res
        fy = 0.5 / np.tan(FoVy / 2) * self.cfg.im_res

        K = np.array([[fx, 0, self.cfg.im_res / 2],
                      [0, fy, self.cfg.im_res / 2],
                      [0, 0, 1]])

        Ks_dict = {0: K}

        return Ks_dict

    def _generate_3d_points(self):
        return sample_3d_points(self.cfg.num_points, self.cfg.scene_size)

    def _create_images(self):
        images = []

        rgb_normalized = np.array(self.cfg.points_rgb) / 255.0
        print(rgb_normalized)

        for cam2world, img_path in zip(self.cam2worlds, self.image_paths):
            points_2d = project_points_3d_to_2d(self.points, self.Ks_dict[0], cam2world)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(points_2d[:, 0], points_2d[:, 1], s=self.cfg.points_size, c=rgb_normalized)
            ax.set_xlim(0, self.cfg.im_res)
            ax.set_ylim(0, self.cfg.im_res)
            ax.invert_yaxis()
            ax.axis("off")

            if self.cfg.save_imgs:
                fig.savefig(img_path, bbox_inches="tight", pad_inches=0)

            images.append(fig)

            plt.close(fig)

        return images

class SyntheticDataset:
    """A simple dataset class."""

    def __init__(
            self,
            parser: SyntheticParser,
            split: str = "train",
            ):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_paths))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        cam2worlds = self.parser.cam2worlds[index]

        data = {
                "K": torch.from_numpy(K).float(),
                "cam2world": torch.from_numpy(cam2worlds).float(),
                "image": torch.from_numpy(image).float(),
                "image_id": item,  # the index of the image in the dataset
                }

        return data

