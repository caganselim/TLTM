import os.path
from detectron2.engine import default_setup
from maskgnn.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from maskgnn_utils.evaluators.coco_helper import load_coco_json

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
        
    # Set score_threshold for builtin models
    thresh = cfg.MASKGNN.SCORE_THRESH_TEST
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = thresh

    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def set_datasets(cfg):

    dataset_root = cfg.DATASETS.DATASET_DIR
    dataset_name =  cfg.DATASETS.DATASET_NAME

    trn_json_file = cfg.DATASETS.TRN_JSON_PATH
    val_json_file = cfg.DATASETS.VAL_JSON_PATH

    trn_loader_mode = cfg.DATASETS.TRN_LOADER_MODE
    val_loader_mode = cfg.DATASETS.VAL_LOADER_MODE

    f_trn, f_val = "train", "test"

    # Set train dataset.
    if trn_loader_mode == "single":

        """
        Can be used to pretrain our model.
        """

        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_train",
                                lambda: load_coco_json(trn_json_file, os.path.join(dataset_root, "train2017"),
                                                       f"{cfg.DATASETS.DATASET_NAME}_train"))

    elif trn_loader_mode == "double":

        """
        Standard train loader.
        """

        from maskgnn_utils.dataset.prep.prep_double_dataset import get_dset_dict_double as get_dset_dict_trn
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_train",
                                lambda f_trn=f_trn: get_dset_dict_trn(dataset_root=dataset_root,
                                                                      dataset_name=dataset_name,
                                                                      json_file=trn_json_file,
                                                                      include_last=False,
                                                                      is_train=True))
    elif trn_loader_mode == "joint":

        """
        This does hallucinated motion + YoutubeVIS frame joint training.
        """

        ytvis_json_file = cfg.DATASETS.JOINT_TRN_VID_JSON
        coco_json_file = cfg.DATASETS.JOINT_TRN_IMG_JSON
        padded_sampling = cfg.DATASETS.PADDED_SAMPLING

        from maskgnn_utils.dataset.prep.prep_joint_dataset import get_dset_joint as get_dset_dict_trn
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_train",
                                lambda f_trn=f_trn: get_dset_dict_trn(dataset_root=dataset_root,
                                                                      ytvis_json_file=ytvis_json_file,
                                                                      coco_json_file=coco_json_file,
                                                                      include_last=False,
                                                                      is_train=True,
                                                                      is_padded=padded_sampling))

    else:
        print("Invalid train data loader mode.")
        raise NotImplementedError

    # Set validation dataset.
    if val_loader_mode == "single":

        from maskgnn_utils.dataset.prep.prep_single_dataset import get_dset_dict_single as get_dset_dict_val
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_test",
                                lambda f_val=f_val: get_dset_dict_val(dataset_root=dataset_root,
                                                                      dataset_name=dataset_name,
                                                                      json_file=val_json_file,
                                                                      include_last=True,
                                                                      is_train=False))

    else:

        from maskgnn_utils.dataset.prep.prep_double_dataset import get_dset_dict_double as get_dset_dict_val
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_test",
                                lambda f_val=f_val: get_dset_dict_val(dataset_root=dataset_root,
                                                                      dataset_name=dataset_name,
                                                                      json_file=val_json_file,
                                                                      include_last=True,
                                                                      is_train=False))

    """
    Set classes
    """

    if cfg.MODEL.FCOS.NUM_CLASSES == 1:

        MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=["object"])
        MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=["object"])

    else:

        if cfg.DATASETS.DATASET_NAME == "ytvis":
            thing_classes = ["person", "giant_panda", "lizard", "parrot", "skateboard", "sedan",
                             "ape", "dog","snake", "monkey", "hand", "rabbit", "duck", "cat",
                             "cow", "fish", "train", "horse", "turtle", "bear", "motorbike",
                             "giraffe", "leopard", "fox", "deer", "owl", "surfboard", "airplane",
                             "truck", "zebra", "tiger", "elephant","snowboard", "boat", "shark",
                             "mouse", "frog", "eagle", "earless_seal", "tennis_racket"]

            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=thing_classes)
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=thing_classes)

        elif cfg.DATASETS.DATASET_NAME == "kitti_mots":

            thing_classes = ["person", "car"]
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=thing_classes)
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=thing_classes)


        elif cfg.DATASETS.DATASET_NAME == "det100":

            thing_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                             'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                             'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                             'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                             'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                             'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                             'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                             'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                             'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'giant panda',
                             'lizard', 'parrot', 'monkey/ape', 'snake', 'hand', 'rabbit', 'duck', 'fish', 'turtle',
                             'leopard', 'fox', 'deer', 'owl', 'tiger', 'shark', 'mouse', 'frog', 'eagle', 'earless_seal']

            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=thing_classes)
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=thing_classes)

    # Eval type: coco, two_frame_tracking, uvos_writer
    MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(evaluator_type=cfg.MASKGNN.EVAL_TYPE)
    MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(evaluator_type=cfg.MASKGNN.EVAL_TYPE)