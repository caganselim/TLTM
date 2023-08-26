import os.path

from detectron2.data.build import filter_images_with_only_crowd_annotations

from maskgnn_utils.dataset.prep.load_multi_coco import load_multi_coco_json
from maskgnn_utils.dataset.prep.prep_double_dataset import get_dset_dict_double
from maskgnn_utils.dataset.prep.prep_double_padded_dataset import get_padded_dset_dict_double
from maskgnn_utils.evaluators.coco_helper import load_coco_json
import logging

def get_dset_joint(dataset_root, ytvis_json_file, coco_json_file, include_last=False, is_train=True, is_padded=False):

    logger = logging.getLogger(__name__)
    logger.info(f"[MaskGNN] - Building joint dataset. Padding is {is_padded}")

    if is_padded:
        ytvis_dicts = get_padded_dset_dict_double(dataset_root, "ytvis_joint", ytvis_json_file, include_last=include_last, is_train=is_train)
    else:
        ytvis_dicts = get_dset_dict_double(dataset_root, "ytvis_joint", ytvis_json_file, include_last=include_last, is_train=is_train)

    #coco_dicts = load_coco_json(coco_json_file, os.path.join(dataset_root, "coco2017train"))

    pth = os.path.join(dataset_root)
    print(dataset_root)

    print(coco_json_file)
    coco_dicts = load_multi_coco_json(coco_json_file, image_root=dataset_root)


    coco_dicts = filter_images_with_only_crowd_annotations(coco_dicts)

    print("Number of ytvis dicts: ", len(ytvis_dicts))
    
    dataset_dicts = ytvis_dicts
    dataset_dicts.extend(coco_dicts)

    logger = logging.getLogger(__name__)
    logger.info("[MaskGNN] - Building joint dataset.")
    
    print("Number of coco dicts: ", len(coco_dicts))
    print("Total number of dicts: ", len(dataset_dicts))
    return dataset_dicts