import os
import json
from detectron2.structures import BoxMode
import time


def get_padded_dset_dict_double(dataset_root, dataset_name, json_file,
                                include_last=False, is_train=True,
                                pad_range=3):

    print("===========================================================")

    # Initial setup
    t0 = time.time()
    dataset_dicts = []
    image_id = 0

    video_pth_prefix = ""
    if dataset_name == "DAVIS":
        video_pth_prefix = "JPEGImages/480p"
    elif dataset_name == "ytvis":
        video_pth_prefix = "JPEGImages"
        if is_train:
            dataset_root = os.path.join(dataset_root, "train")
        else:
            dataset_root = os.path.join(dataset_root, "valid")
    elif dataset_name == "kitti_mots":
        video_pth_prefix = "image_02"
    elif dataset_name == "ytvis_joint":
        video_pth_prefix = "ytvis"

    with open(json_file) as f:

        print("Dataset root: ", dataset_root)
        print("Opened json file: ", json_file)

        # Load YoutubeVIS dict.
        dset_json = json.load(f)

        # Videos
        for idx, vid_dict in enumerate(dset_json["videos"]):

            seq_len = len(vid_dict["frames"])
            seq_len = seq_len if include_last else seq_len-1

            if "video_id" in vid_dict.keys():
                video_id = vid_dict["video_id"]
            else:
                video_id = -1

            for i in range(seq_len):

                # Process frame dictionaries. First; save the current one.
                frame_dict_0 = vid_dict["frames"][i]
                frame_name_0 = frame_dict_0["frame_name"]
                frame_annotations_0 = frame_dict_0["annotations"]

                if len(frame_annotations_0) == 0:
                    print("Skip!")
                    continue

                # Then, perform the following frames. Check the following statements
                num_frames_to_be_saved = pad_range if seq_len - i > pad_range else seq_len - i
                next_frame_infos = []

                assert num_frames_to_be_saved > 0, "num_frames_to_be_saved must be higher than 0"
                assert num_frames_to_be_saved <= pad_range, "num_frames_to_be_saved must be less than or equal to pad_range"

                for k in range(num_frames_to_be_saved):

                    # This will be appended. Prepare an inner record
                    frame_dict_next = vid_dict["frames"][i+k+1]
                    frame_annotations_next = frame_dict_next["annotations"]

                    if len(frame_annotations_next) == 0:
                        print("Skip!")
                        continue

                    frame_name_next = frame_dict_next["frame_name"]

                    inner_rec = {"width": vid_dict["width"], "height": vid_dict["height"],
                                 "file_name_1": os.path.join(dataset_root, video_pth_prefix, frame_name_next),
                                 "annotations_1": frame_annotations_next}

                    # Save in the end
                    next_frame_infos.append(inner_rec)

                if len(next_frame_infos) == 0:
                    print("important skip")
                    continue

                # Process objects and save it
                # Create a record dict to store info for the consecutive frames
                record = {"width": vid_dict["width"], "height": vid_dict["height"],
                          "file_name_0": os.path.join(dataset_root, video_pth_prefix, frame_name_0),
                          "annotations_0": frame_annotations_0,
                          "next_frame_dicts": next_frame_infos,
                          "image_id": image_id, "video_id": video_id}

                # Increment image id.
                image_id += 1

                # Save the record
                dataset_dicts.append(record)

    print(f"Elapsed: {time.time() - t0} seconds")

    return dataset_dicts


if __name__ == "__main__":



    ytvis_dicts = get_padded_dset_dict_double("datasets/det100/frames", "ytvis_joint",
                                              "datasets/jsons/ytvis/ytvis_train.json")

    print(len(ytvis_dicts))

    first_dict = ytvis_dicts[0]
    print("First dict keys: ", first_dict.keys())

    print(first_dict)