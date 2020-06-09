import os
import sys
import json


# {7: 384, 8: 443, 6: 186}

def check_class_bbox_count(ann_info):
    class_ann = {}
    for ann in ann_info['annotations']:
        cid = ann['category_id']
        if cid not in class_ann:
            class_ann[cid] = 1
        else:
            class_ann[cid] += 1
    print(class_ann)
    return class_ann


def main(ann_json_path, save_name, dev_per=0.1):
    dev_per = float(dev_per)
    with open(ann_json_path, 'r') as f:
        ann_info = json.load(f)

    class_ann_cnt = check_class_bbox_count(ann_info)
    class_ann_cnt_test = {k: round(v * dev_per) for k, v in class_ann_cnt.items()}
    print(class_ann_cnt_test)

    ann_test_part = []
    img_test_id_part = set([])
    ann_train_part = []
    img_train_id_part = set([])

    current_class_ann_cnt = {}
    for ann in ann_info['annotations']:

        cid = ann['category_id']
        all_cnt = class_ann_cnt_test[cid]

        if cid in current_class_ann_cnt:
            cur_cnt = current_class_ann_cnt[cid]
            if cur_cnt > all_cnt:
                if ann['image_id'] not in img_test_id_part:
                    if ann['image_id'] == 556:
                        print(111, ann, cur_cnt, all_cnt)
                    ann_train_part.append(ann)
                    img_train_id_part.add(ann['image_id'])
                else:
                    if ann['image_id'] == 556:
                        print(222, ann, cur_cnt, all_cnt)
                    ann_test_part.append(ann)
                    img_test_id_part.add(ann['image_id'])
            else:
                if ann['image_id'] in img_train_id_part:
                    if ann['image_id'] == 556:
                        print(333, ann, cur_cnt, all_cnt)
                    ann_train_part.append(ann)
                    img_train_id_part.add(ann['image_id'])
                else:
                    if ann['image_id'] == 556:
                        print(444, ann, cur_cnt, all_cnt)
                    ann_test_part.append(ann)
                    img_test_id_part.add(ann['image_id'])
            current_class_ann_cnt[cid] += 1
        else:
            if ann['image_id'] not in img_train_id_part:
                if ann['image_id'] == 556:
                    print(555, ann, cur_cnt, all_cnt)
                current_class_ann_cnt[cid] = 1
                ann_test_part.append(ann)
                img_test_id_part.add(ann['image_id'])
            else:
                if ann['image_id'] == 556:
                    print(666, ann, cur_cnt, all_cnt)
                ann_train_part.append(ann)
                img_train_id_part.add(ann['image_id'])

    img_test_part = []
    img_train_part = []
    for img in ann_info['images']:
        if img['id'] in img_test_id_part:
            img_test_part.append(img)
        elif img['id'] in img_train_id_part:
            img_train_part.append(img)
        else:
            print(img)

    # ids_with_ann1 = set(_['image_id'] for _ in ann_test_part)
    # print(ids_with_ann1-img_test_id_part)
    # print(img_test_id_part-ids_with_ann1)
    # ids_with_ann2 = set(_['image_id'] for _ in ann_train_part)
    # print(ids_with_ann2-img_train_id_part)
    # print(img_train_id_part-ids_with_ann2)

    test_ann_info = {
        'annotations': ann_test_part,
        'images': img_test_part,
        'categories': ann_info['categories'],
    }
    print('val', len(ann_test_part), len(img_test_part))
    train_ann_info = {
        'annotations': ann_train_part,
        'images': img_train_part,
        'categories': ann_info['categories'],
    }
    print('train', len(ann_train_part), len(img_train_part))

    ann_dir = os.path.dirname(ann_json_path)
    print(ann_dir)
    with open(os.path.join(ann_dir, '%s_train.json' % save_name), 'w') as f:
        json.dump(train_ann_info, f, sort_keys=True, indent=2)
    with open(os.path.join(ann_dir, '%s_val.json' % save_name), 'w') as f:
        json.dump(test_ann_info, f, sort_keys=True, indent=2)


main('data/train/annotations/train.json', 'annotations_split_0.15', 0.15)
print('已完成数据集的划分！')