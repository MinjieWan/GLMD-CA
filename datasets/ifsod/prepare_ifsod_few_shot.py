import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = "./annotations/all.json"
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data["categories"]:
        new_all_cats.append(cat)

    # 以图像编号索引图像信息
    id2img = {}
    for i in data["images"]:
        id2img[i["id"]] = i

    # 将每个类别的annotations划分出来, 并以类别编号为索引
    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        # 以类别编号为索引
        for c in ID2CLASS.keys():
            img_ids = {}
            # 遍历所有属于该类别的annotations
            # 将属于同一图像的annotations划分到一起
            # 以图像编号为索引
            for a in anno[c]:
                if a["image_id"] in img_ids:
                    img_ids[a["image_id"]].append(a)
                else:
                    img_ids[a["image_id"]] = [a]

            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10]:
                while True:
                    # 随机采样shots个图像(一个图像中不一定只包含一个annotation)
                    imgs = random.sample(list(img_ids.keys()), shots) 
                    for img in imgs:
                        # 如果该图像编号已经被使用, 则跳过
                        skip = False
                        for s in sample_shots:
                            if img == s["image_id"]:
                                skip = True
                                break
                        if skip:
                            continue

                        # 如果已经采样的annotations数目 + 当前图像中annotations数目 > 设定的shots数量
                        # 跳过该图像
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue

                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])

                        # 如果采样到指定shots数量的annotations, 则跳出循环
                        if len(sample_shots) == shots:
                            break
                    # 如果采样到指定shots数量的annotations, 则跳出循环
                    if len(sample_shots) == shots:
                        break
                    
                new_data = {
                    "info": data["info"],
                    "licenses": data["licenses"],
                    "images": sample_imgs,
                    "annotations": sample_shots,
                }
                save_path = get_save_path_seeds(
                    data_path, ID2CLASS[c], shots, i
                )
                new_data["categories"] = new_all_cats
                with open(save_path, "w") as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    save_dir = os.path.join(".", "ifsodsplit")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path


if __name__ == "__main__":
    ID2CLASS = {
        1: "person",
        2: "car",
        3: "bicycle",
        4: "uav",
        5: "motorcycle",
        6: "truck",
        7: "bus",
        8: "bison",
        9: "deer",
        10: "dog",
        11: "boat"
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
