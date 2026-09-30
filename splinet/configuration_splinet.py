from transformers import FNetConfig


class SpliNetConfig(FNetConfig):
    """Configuration for the final SpliNet architecture used in this repository.

    The public model is intentionally fixed to the selected spline mixer:
    order-2 cardinal B-spline, single-sided sequence mixing, radius 16.
    Other historical spline orders/sidedness variants are not exposed here.
    """

    model_type = "splinet"

    def __init__(
        self,
        splinet_num_heads=12,
        splinet_radius=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.splinet_num_heads = int(splinet_num_heads)
        self.splinet_radius = int(splinet_radius)
        self.splinet_order = 2
        self.splinet_sidedness = "single"

        if self.hidden_size % self.splinet_num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"splinet_num_heads={self.splinet_num_heads}"
            )
        if self.splinet_radius < 1:
            raise ValueError("splinet_radius must be positive")
