import os
import json
from detectron2.structures import BoxMode
import time

def get_dset_dict_double(dataset_root, dataset_name, json_file, include_last=False, is_train=True):

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

            # if idx % 500 == 0:
            #     print("Processing video: ", idx)

            seq_len = len(vid_dict["frames"])
            seq_len = seq_len if include_last else seq_len-1

            if "video_id" in vid_dict.keys():

                video_id = vid_dict["video_id"]
            else:
                video_id = -1

            for i in range(seq_len):

                if include_last and i == (seq_len - 1):

                    # Handle the last frame.
                    frame_dict_0 = vid_dict["frames"][i]
                    frame_name_0 = frame_dict_0["frame_name"]

                    # Create a record dict to store info for the consecutive frames
                    record = {"width": vid_dict["width"], "height": vid_dict["height"],
                              "file_name_0": os.path.join(dataset_root, video_pth_prefix, frame_name_0),
                              "image_id": image_id, "video_id":video_id}

                    if "annotations" in frame_dict_0.keys():
                        frame_annotations_0 = frame_dict_0["annotations"]

                        print("prev frame_annot: " , frame_annotations_0)
                        # Class-ids - 1
                        # for idx, annot in enumerate(frame_annotations_0):
                        #     annot["category_id"] = annot["category_id"] -1

                        print("next frame_annot: " , frame_annotations_0)

                        record["annotations_0"] = frame_annotations_0

                    # Increment image id.
                    image_id += 1
                    dataset_dicts.append(record)

                else:

                    # Get frame dictionaries
                    frame_dict_0, frame_dict_1 = vid_dict["frames"][i], vid_dict["frames"][i+1]
                    frame_name_0, frame_name_1 = frame_dict_0["frame_name"], frame_dict_1["frame_name"]

                    # Create a record dict to store info for the consecutive frames
                    record = {"width": vid_dict["width"], "height": vid_dict["height"],
                              "file_name_0": os.path.join(dataset_root, video_pth_prefix, frame_name_0),
                              "file_name_1": os.path.join(dataset_root, video_pth_prefix, frame_name_1),
                              "image_id": image_id, "video_id":video_id}

                    if "annotations" in frame_dict_1.keys():
                        frame_annotations_0 = frame_dict_0["annotations"]
                        frame_annotations_1 = frame_dict_1["annotations"]

                        if (len(frame_annotations_0) == 0 or len(frame_annotations_1) == 0) and is_train:
                            #print("Skipping empty frames.")
                            continue
                        else:

                            # Class-ids - 1
                            # for annot in frame_annotations_0:
                            #     annot["category_id"] = annot["category_id"] - 1
                            #
                            # # # Class-ids - 1
                            # for annot in frame_annotations_1:
                            #     annot["category_id"] = annot["category_id"] - 1


                            # Process objects and save it
                            record["annotations_0"] = frame_annotations_0
                            record["annotations_1"] = frame_annotations_1

                            # Increment image id.
                            image_id += 1
                            # Save the record
                            dataset_dicts.append(record)

    print(f"Elapsed: {time.time() - t0} seconds")

    return dataset_dicts


if __name__ == "__main__":

    c = get_dset_dict_double(dataset_root="./YoutubeVIS", dataset_name="ytvis",
                             json_file="../../datasets/jsons/ytvis_train_one_video_v10.json") #- test passed

    print(c)