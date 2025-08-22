import json
import os
from pycocotools.coco import COCO


FILTER_CAT_ID = {1, 3, 2, 4, 8, 6, 18, 9}

COCO_2017_TRAIN_JSON = "../COCO2017/annotations/instances_train2017.json"

GCOCO_ANNS_PATH = "./annotations/"
os.makedirs(GCOCO_ANNS_PATH, exist_ok=True)
GCOCO_TRAIN_JSON = os.path.join(GCOCO_ANNS_PATH, "train.json")


def main():
    coco_train_anns = COCO(COCO_2017_TRAIN_JSON)
    coco_anns = coco_train_anns.anns
    
    gcoco_train_anns = {"info": coco_train_anns.dataset["info"],
                        "licenses": coco_train_anns.dataset["licenses"],
                        "images": coco_train_anns.dataset["images"],
                        "annotations": [],
                        "categories": coco_train_anns.dataset["categories"]}
    
    # filter COCO anns from categories overlapped with IFSOD
    for ann_info in coco_anns.values():
        if ann_info["category_id"] in FILTER_CAT_ID:
            continue
        gcoco_train_anns["annotations"].append(ann_info)

    with open(GCOCO_TRAIN_JSON, "w") as f:
        json.dump(gcoco_train_anns, f)
    

if __name__ == "__main__":
    main()
