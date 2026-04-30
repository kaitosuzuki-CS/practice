import torch


class Crop:
    def __init__(self, hps):
        self.output_size = hps.output_size

    def random_crop(self, x):
        B, C, H, W = x.shape

        crop_max = H - self.output_size + 1

        w1 = torch.randint(0, crop_max, (B,), device=x.device)
        h1 = torch.randint(0, crop_max, (B,), device=x.device)

        cropped = torch.empty(
            (B, C, self.output_size, self.output_size), dtype=x.dtype, device=x.device
        )

        for i, (img, w_start, h_start) in enumerate(zip(x, w1, h1)):
            cropped[i] = img[
                :,
                h_start : h_start + self.output_size,
                w_start : w_start + self.output_size,
            ]

        return cropped

    def center_crop(self, x):
        h, w = x.shape[-2:]
        start_h = (h - self.output_size) // 2
        start_w = (w - self.output_size) // 2
        return x[
            :,
            :,
            start_h : start_h + self.output_size,
            start_w : start_w + self.output_size,
        ]
