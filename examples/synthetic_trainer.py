import time
import math
import torch
import tyro
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
from datasets.synthetic import SyntheticDataset, SyntheticParser
from simple_trainer import (
        Config,
        Runner,
        )

@dataclass
class SyntheticConfig(Config):
    # Synthetic scene parameters
    num_imgs: int = 10
    im_res: int = 128
    white_background: bool = True

    num_gaussians: int = 1
    scene_size: float = 2.5
    num_points: int = 5
    points_size: int = 5
    points_rgb: List = field(default_factory=lambda: [0, 0, 255])
    radius: float = 10

    save_imgs: bool = True

class SyntheticRunner(Runner):
    """Engine for training and testing."""

    def __init__(self, cfg: SyntheticConfig) -> None:
        super().__init__(cfg=cfg)
        self.cfg = cfg

    def _load_data(self):
        # Load data: Training data should contain initial points and colors.
        self.parser = SyntheticParser(
            cfg=cfg,
            normalize=True,
        )
        self.trainset = SyntheticDataset(
            self.parser,
            split="train",
        )
        self.valset = SyntheticDataset(self.parser, split="val")

    def _init_model(self, feature_dim):
        self.splats, self.optimizers = self._init_splats(
            self.cfg.num_gaussians,
            np.array(self.cfg.points_rgb) / 255.0,
            scene_size=self.cfg.scene_size,
            scene_scale=self.scene_scale,
            sh_degree=self.cfg.sh_degree,
            init_opacity=self.cfg.init_opa,
            sparse_grad=self.cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim if cfg.app_opt else None,
            device=self.device,
        )
        print("Model initialized. Number of GS:", self.cfg.num_gaussians)

    def _init_splats(
        self,
        N: int = 1,  
        rgb: npt.NDArray[Any] = np.array([0, 0, 1.0]),
        scene_size: float = 2.5,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        sparse_grad: bool = False,
        batch_size: int = 1,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
    ) -> Tuple[torch.nn.ParameterDict, List[torch.optim.Adam | torch.optim.SparseAdam]]:

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        means3d = torch.randn(N, 3) * scene_size - scene_size / 2 # [N, 3] in [-scene_size / 2, scene_size / 2]
        scales = torch.log(torch.rand(N, 3)) + 0.1 # [N, 3] in [log(0.1), log(1.1)]
        quats = torch.rand((N, 4))  # [N, 4] in [0, 1]
        opacities = torch.logit(torch.full((N,), init_opacity)) # [N,]

        params = [
            # name, value, lr
            ("means3d", torch.nn.Parameter(means3d), 1.6e-4 * scene_scale),
            ("scales", torch.nn.Parameter(scales), 5e-3),
            ("quats", torch.nn.Parameter(quats), 1e-3),
            ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ]

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.rand(N, (sh_degree + 1) ** 2, 3) - 0.5  # [N, K, 3] in [-0.5, 0.5]
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
        else:
            # features will be used for appearance and view-dependent shading
            features = torch.rand(N, feature_dim)  # [N, feature_dim]
            params.append(("features", torch.nn.Parameter(features), 2.5e-3))
            colors = torch.logit(torch.from_numpy(rgb).repeat(N, 1))  # [N, 3]
            params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1
        optimizers = [
            (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
                [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        ]
        return splats, optimizers


def main(cfg: SyntheticConfig):
    runner = SyntheticRunner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(SyntheticConfig)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
