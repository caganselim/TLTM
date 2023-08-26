import torch.nn
from torch import nn
from maskgnn.modeling.obj_encoder.build import OBJ_ENCODER_REGISTRY

@OBJ_ENCODER_REGISTRY.register()
class EncoderCNN(nn.Module):

    """MLP encoder, maps segmentation to latent state."""

    def __init__(self, cfg):

        super(EncoderCNN, self).__init__()

        self.obj_pooling_src = cfg.MASKGNN.NODE_REP.POOL_SRC
        pool_dim = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION

        if self.obj_pooling_src == "cls_ctr":
            input_dim = cfg.MODEL.FCOS.NUM_CLASSES
        else:
            # backbone case. => 256
            input_dim = cfg.MODEL.OBJ_ENCODER.ENCODER_CNN.INPUT_DIM

        linear_in = (pool_dim // 2) *(pool_dim // 2)
        hidden_dim = cfg.MODEL.OBJ_ENCODER.ENCODER_CNN.HIDDEN_DIM
        output_dim = cfg.MODEL.OBJ_ENCODER.OUTPUT_DIM

        # before: [N, 256, 14, 14], after: [N, 64, 14, 14]
        self.cn1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)

        # before: [N, 64, 14, 14], after: [N, 64, 7, 7]
        self.mp =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # before: [N, 64, 7, 7], after: [N, 16, 7, 7]
        self.cn2 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, stride=1, padding=1, bias=True)

        # before: [N, 16, 7, 7], after: [N, 1, 7, 7]
        self.cn3 = nn.Conv2d(hidden_dim // 4, hidden_dim // 16, kernel_size=3, stride=1, padding=1, bias=True)

        # before: [N, 49], after: [N, 8]
        self.fc = nn.Linear(linear_in*(hidden_dim // 16), output_dim)

        self.act = nn.ReLU()

    def forward(self, ins):

        # before: [N, 256, 14, 14], after: [N, 64, 14, 14]
        h = self.mp(self.act(self.cn1(ins)))

        # before: [N, 64, 14, 14], after: [N, 64, 7, 7]
        h = self.act(self.cn2(h))

        # before: [N, 16, 7, 7], after: [N, 1, 7, 7]
        h = self.act(self.cn3(h))

        # before: [N, 16, 7, 7], after: [N, 1, 7, 7]
        h = h.view(h.shape[0], -1)

        # before: [N, 49], after: [N, 8]
        h = self.fc(h)

        return h


