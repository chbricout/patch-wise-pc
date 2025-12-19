import torch


class Rgb2GenT:
    """Genaro's version of YCoCg-R"""

    def __call__(self, rgb_images: torch.Tensor):
        assert rgb_images.size(0) == 3

        def forward_lift(x, y):
            diff = (y - x) % 256
            average = (x + (diff >> 1)) % 256
            return average, diff

        red, green, blue = rgb_images[0, ...], rgb_images[1, ...], rgb_images[2, ...]
        temp, co = forward_lift(red, blue)
        y, cg = forward_lift(green, temp)
        ycc_images = torch.stack([y, co, cg], dim=0)
        return ycc_images


class Rgb2BenT:
    """Benjie's version of YCoCg-R"""

    def __init__(self, channel_last=False, rgb_range=(-1, 1)):
        self.channel_last = channel_last
        self.rgb_range = rgb_range

    def __call__(self, x):
        if self.channel_last:
            R, G, B = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        else:
            R, G, B = x[0, :, :], x[1, :, :], x[2, :, :]

        R = (
            (R - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0]) * 255
        ).long()
        G = (
            (G - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0]) * 255
        ).long()
        B = (
            (B - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0]) * 255
        ).long()

        Co = R - B
        tmp = B + Co // 2
        Cg = G - tmp
        Y = tmp + Cg // 2

        Co += 256
        Cg += 256

        if self.channel_last:
            return torch.stack((Y, Co, Cg), dim=2)
        else:
            return torch.stack((Y, Co, Cg), dim=0)


class Ycc2RgbLossless:
    def __call__(self, ycc_images):
        assert ycc_images.size(0) == 3

        def reverse_lift(average, diff):
            x = (average - (diff >> 1)) % 256
            y = (x + diff) % 256
            return x, y

        y, co, cg = ycc_images[0, ...], ycc_images[1, ...], ycc_images[2, ...]
        green, temp = reverse_lift(y, cg)
        red, blue = reverse_lift(temp, co)
        rgb_images = torch.stack([red, green, blue], dim=0)
        return rgb_images


class Rgb2YccLossy:
    def __call__(self, rgb_images: torch.Tensor):
        assert rgb_images.size(0) == 3

        dequantized_images = (rgb_images / 127.5) - 1
        red, green, blue = (
            dequantized_images[0, ...],
            dequantized_images[1, ...],
            dequantized_images[2, ...],
        )

        red = (red + 1) / 2
        green = (green + 1) / 2
        blue = (blue + 1) / 2

        Co = red - blue
        tmp = blue + Co / 2
        Cg = green - tmp
        Y = tmp + Cg / 2
        Y = Y * 2 - 1

        transformed_images = torch.stack((Y, Co, Cg), dim=0)
        return torch.floor(((transformed_images + 1) / 2) * 256).long().clip(0, 255)


class Ycc2Rgb2Lossy:
    def __call__(self, ycc_images: torch.Tensor):
        assert ycc_images.size(-1) == 3

        dequantized_images = (ycc_images / 127.5) - 1
        Y, Co, Cg = (
            dequantized_images[..., 0],
            dequantized_images[..., 1],
            dequantized_images[..., 2],
        )

        # Convert the range of Y back to [0, 1]
        Y = (Y + 1) / 2

        tmp = Y - Cg / 2
        G = Cg + tmp
        B = tmp - Co / 2
        R = B + Co

        transformed_images = torch.stack((R, G, B), dim=-1)
        rgb_images = (transformed_images * 255).long().clip(0, 255).to(torch.uint8)
        return rgb_images
