import torch
import torch.nn as nn

# pip install torchinfo torchview
from torchinfo import summary
from torchview import draw_graph


class SpecEncoder2D(nn.Module):
    def __init__(self, in_ch: int = 1, enc_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, enc_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,F,T) -> (B,enc_ch,F',T')
        return self.net(x)


if __name__ == "__main__":
    model = SpecEncoder2D(in_ch=1, enc_ch=128)

    # Example input shape: (batch, channels, freq, time)
    input_size = (1, 1, 45, 297)

    print("=" * 80)
    print("MODEL SUMMARY WITH TORCHINFO")
    print("=" * 80)

    summary(
        model,
        input_size=input_size,
        col_names=("input_size", "output_size", "num_params", "kernel_size"),
        depth=4,
        verbose=1,
    )

    print("\n" + "=" * 80)
    print("ARCHITECTURE DIAGRAM WITH TORCHVIEW")
    print("=" * 80)

    graph = draw_graph(
        model,
        input_size=input_size,
        expand_nested=True,
        depth=4,
        graph_name="SpecEncoder2D",
        save_graph=True,
        filename="spec_encoder2d_architecture",
        directory=".",
    )

    # In notebooks, this often displays directly:
    graph.visual_graph

    print("\nDiagram saved as: spec_encoder2d_architecture.png")