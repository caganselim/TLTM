# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch

from detectron2.evaluation import  verify_results
from detectron2.checkpoint import DetectionCheckpointer

from maskgnn_utils.dataset.dataset_setup import setup, set_datasets
from maskgnn_utils.trainer import MaskGNNTrainer

def main(args):

    cfg = setup(args)
    set_datasets(cfg)
    eval = args.eval_only

    if eval:

        model = MaskGNNTrainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = MaskGNNTrainer.test(cfg, model)

        if comm.is_main_process():

            verify_results(cfg, res)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """

    trainer = MaskGNNTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
