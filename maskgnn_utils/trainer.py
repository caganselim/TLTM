import logging

import os

import torch
from detectron2.config import configurable
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluators
from maskgnn_utils.evaluators import TrackingDeltaEvaluator, UVOSWriter, COCOEvaluatorVID, DebugWriter, DebugWriterCocoMotion

from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, DatasetMapper, DatasetFromList, MapDataset, \
    build_batch_data_loader

from detectron2.data import (
    build_detection_test_loader,
)

from maskgnn_utils.evaluators.kitti_mots_writer import KITTIMOTSWriter
from maskgnn_utils.evaluators.ytvis_writer import YTVISWriter


def _train_loader_from_config_v2(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        # sampler_name = cfg.INPUT.DATALOADER.SAMPLER_TRAIN
        if cfg.DATASETS.TRN_LOADER_MODE == "joint":
            sampler_name = "JointSampler"
        else:
            sampler_name = "TrainingSampler"
        print("Inside new train loader from config! Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)

        elif sampler_name == "JointSampler":
            repeat_factors = torch.Tensor([8.0 if "video_id" in dataset_dict else 1.0 for dataset_dict in dataset])
            sampler = RepeatFactorTrainingSampler(repeat_factors)

        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config_v2)
def build_detection_train_loader_v2(
        dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


########################################################################################################################
########################################################################################################################
########################################################################################################################

class MaskGNNTrainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    # self.iter => trainer base current iteration.


    """

    @classmethod
    def build_train_loader(cls, cfg):

        # Overrided this function to decide a custom dataset mapper for our setup.
        # To pretrain our model with static images only, we use the default DatasetMapper
        # of detectron2. For the trainings that we require consecutive images, we use two_frame_mapper.
        # Plus, we support joint training.

        print("Building train loader!")

        if cfg.DATASETS.TRN_LOADER_MODE == "double":
            from maskgnn_utils.dataset.mappers.two_frame_mapper import TwoFrameDatasetMapperTrain as DatasetMapper

        elif cfg.DATASETS.TRN_LOADER_MODE == "joint":
            print("JOINT-MAPPER")
            from maskgnn_utils.dataset.mappers.joint_mapper import JointMapper as DatasetMapper

        else:
            from maskgnn_utils.dataset.mappers.one_frame_mapper import DatasetMapper


        mapper = DatasetMapper(cfg, is_train=True)
        #sampler = cfg.INPUT.DATALOADER.SAMPLER_TRAIN
        # Sampler modification goes here.

        return build_detection_train_loader_v2(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        print(f"Building test loader!")

        if cfg.DATASETS.VAL_LOADER_MODE == "double":
            from maskgnn_utils.dataset.mappers.two_frame_mapper import TwoFrameDatasetMapperTest as DatasetMapper
        else:
            from maskgnn_utils.dataset.mappers.one_frame_mapper import OneFrameDatasetMapper as DatasetMapper

        mapper = DatasetMapper(cfg, is_train=False)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """

        # Prepare the output folder.
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            print("Evaluating last checkpoint: ", output_folder)

            # If the output folder doesn't exist, create.
            os.makedirs(output_folder, exist_ok=True)

        evaluator_list = []

        # Get the evaluator type from our dataset settings.
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco_vid"]:
            evaluator_list.append(COCOEvaluatorVID(dataset_name, output_dir=output_folder))

        # MaskGNN - Add two frame tracking evaluator type.
        if evaluator_type in ["two_frame_tracking"]:
            evaluator_list.append(TrackingDeltaEvaluator(dataset_name, output_dir=output_folder))

        if evaluator_type in ["uvos_writer"]:
            evaluator_list.append(UVOSWriter(dataset_name, output_dir=output_folder))

        if evaluator_type in ["ytvis_writer"]:
            evaluator_list.append(YTVISWriter(dataset_name, output_dir=output_folder))

        if evaluator_type in ["debug_writer"]:
            evaluator_list.append(DebugWriter(dataset_name, output_dir=output_folder))

        if evaluator_type in ["debug_writer_coco_motion"]:
            evaluator_list.append(DebugWriterCocoMotion(dataset_name, output_dir=output_folder))

        if evaluator_type in ["kitti_mots_writer"]:
            evaluator_list.append(KITTIMOTSWriter(dataset_name, output_dir=output_folder))

        # Return the evaluator if there is only one otherwise return DatasetEvaluators Object.
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)





