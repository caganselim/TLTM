# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# NUM CLASSES TRICK
_C.MODEL.ROI_HEADS.NUM_CLASSES = 40
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 40
_C.MODEL.RETINANET.NUM_CLASSES = 40
_C.MODEL.FCOS.NUM_CLASSES = 40

_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'


# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.VOVNET = CN()

_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.VOVNET.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.VOVNET.WITH_MODULATED_DCN = False
_C.MODEL.VOVNET.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# CenterMask
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION = "area"
_C.MODEL.MASKIOU_ON = False
_C.MODEL.MASKIOU_LOSS_WEIGHT = 1.0

_C.MODEL.ROI_MASKIOU_HEAD = CN()
_C.MODEL.ROI_MASKIOU_HEAD.NAME = "MaskIoUHead"
_C.MODEL.ROI_MASKIOU_HEAD.CONV_DIM = 256
_C.MODEL.ROI_MASKIOU_HEAD.NUM_CONV = 4

# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.ROI_KEYPOINT_HEAD.ASSIGN_CRITERION = "ratio"

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #

# -----------------------------<----------------------------------------------- #
# MATCHER
# ---------------------------------------------------------------------------- #
_C.MODEL.MATCHER = CN()
_C.MODEL.MATCHER.NAME = "ObjectMatcher"

_C.MASKGNN = CN()

# -- FORGET MODE -- #
_C.MASKGNN.POSTPROCESSING = CN()
_C.MASKGNN.POSTPROCESSING.FORGET_MODE = False
_C.MASKGNN.POSTPROCESSING.NUM_MISSES_TO_FORGET = 5

# -- POOL SETTINGS -- #
_C.MASKGNN.NODE_REP = CN()
_C.MASKGNN.NODE_REP.POOL_SRC = "backbone"
_C.MASKGNN.NODE_REP.FRAME0_SRC = "gt"
_C.MASKGNN.NODE_REP.FRAME1_SRC = "gt"
_C.MASKGNN.NODE_REP.OBJ_POOL_LEVELS = []
# ------------------- #

_C.MASKGNN.PADDED_TRAINING_ON = False
_C.MASKGNN.FREEZE_CLS_HEADS = False
_C.MASKGNN.EVAL_TYPE= "uvos_writer"
_C.MASKGNN.VIS_TRAIN= False
_C.MASKGNN.SCORE_THRESH_TEST= 0.1

# --- RESFUSER ---#
_C.MODEL.RESFUSER = CN()
_C.MODEL.RESFUSER.MODE = "single"
_C.MODEL.RESFUSER.NAME = "ResFuserV1"
_C.MODEL.RESFUSER.FMAP_DIM = 256
_C.MODEL.RESFUSER.HIDDEN_DIM = 256
# ----------------#

# ---GNN--- #
_C.MODEL.GNN = CN()
_C.MODEL.GNN.IS_ON = True
_C.MODEL.GNN.NAME = "Obj2ObjGNN"
_C.MODEL.GNN.HIDDEN_DIM = 128
_C.MODEL.GNN.ACTION_DIM = 0
_C.MODEL.GNN.MP_ITERS = 1
# ---------- #

# OBJ_ENCODER #
_C.MODEL.OBJ_ENCODER = CN()
_C.MODEL.OBJ_ENCODER.NAME = "EncoderCNN"
_C.MODEL.OBJ_ENCODER.OUTPUT_DIM = 512
_C.MODEL.OBJ_ENCODER.ENCODER_CNN = CN()
_C.MODEL.OBJ_ENCODER.ENCODER_CNN.INPUT_DIM = 256
_C.MODEL.OBJ_ENCODER.ENCODER_CNN.HIDDEN_DIM = 512

# ----------- #

# Datasets
_C.DATASETS.TRN_LOADER_MODE =  "double"
_C.DATASETS.VAL_LOADER_MODE =  "single"
_C.DATASETS.PADDED_SAMPLING =  False
_C.DATASETS.TRN_JSON_PATH = "datasets/jsons/ytvis/ytvis_train.json"
_C.DATASETS.VAL_JSON_PATH = "datasets/jsons/ytvis/ytvis_val.json"
_C.DATASETS.DATASET_DIR = "datasets/det100/frames"
_C.DATASETS.DATASET_NAME = "ytvis"

_C.DATASETS.JOINT_TRN_IMG_JSON = "datasets/det100/coco_openimages_40class.json"
#_C.DATASETS.JOINT_TRN_IMG_JSON = "datasets/jsons/coco/coco_train2017_ytvis_classes.json"
_C.DATASETS.JOINT_TRN_VID_JSON = "datasets/jsons/ytvis/ytvis_train.json"


# COCO_MOTION_AUG Parameters
_C.DATASETS.COCO_MOTION_AUG = CN()
_C.DATASETS.COCO_MOTION_AUG.PERSPECTIVE = False
_C.DATASETS.COCO_MOTION_AUG.AFFINE = True
_C.DATASETS.COCO_MOTION_AUG.BRIGHTNESS_RANGE = (-20, 20)
_C.DATASETS.COCO_MOTION_AUG.HUE_SATURATION_RANGE = (-10, 10)
_C.DATASETS.COCO_MOTION_AUG.PERSPECTIVE_MAGNITUDE = 0.0
_C.DATASETS.COCO_MOTION_AUG.SCALE_RANGE = 1.0
_C.DATASETS.COCO_MOTION_AUG.TRANSLATE_RANGE_X = (-0.20, 0.20)
_C.DATASETS.COCO_MOTION_AUG.TRANSLATE_RANGE_Y = (-0.20, 0.20)
_C.DATASETS.COCO_MOTION_AUG.ROTATION_RANGE = (-20, 20)
_C.DATASETS.COCO_MOTION_AUG.MOTION_BLUR = True
_C.DATASETS.COCO_MOTION_AUG.MOTION_BLUR_KERNEL_SIZES = (7,9)
_C.DATASETS.COCO_MOTION_AUG.MOTION_BLUR_PROB = 0.5
_C.DATASETS.COCO_MOTION_AUG.IDENTITY_MODE = False
_C.DATASETS.COCO_MOTION_AUG.SEED_OVERRIDE = None