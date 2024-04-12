import timm
import torch


class TransferModel(torch.nn.Module):
    # timm モデルをバックボーンとする転移学習モデル
    # ref: https://logmi.jp/tech/articles/325737

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_chans=3,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )
        self.in_features = self.backbone.num_features
        self.out_features = num_classes

        # for timm.data.resolve_data_config
        self.pretrained_cfg = self.backbone.pretrained_cfg

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = torch.nn.Linear(
            self.in_features,
            self.out_features,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

    def set_backbone_requires_grad(self, value):
        for p in self.backbone.parameters():
            p.requires_grad = False
