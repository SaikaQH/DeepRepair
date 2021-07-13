'''
    --- undone yet ---
    how to deal with failure cases
    from generating by pre-trained model
    to emsembling re-fine dataset
'''

from config import FailureCaseConfig, DatasetConfig

import os
import sklearn
import scipy
import cv2
import numpy as np

from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale

fail_config = FailureCaseConfig
data_config = DatasetConfig

""" ---- divide train/test data ---- """
def divide_train_test_data(fail_config, data_config):
    lbl_file = os.path.join(data_config.cifar10c_path, "labels.npy")
    lbl_list = np.load(lbl_file)

    error_list_dir = os.path.join(
            fail_config.fail_img_dir, 
            "error_data/error_list_from_normal_{}_divide".format(fail_config.model_type))

    error_divide_list_save_dir = error_list_dir
    if not os.path.exists(error_divide_list_save_dir):
        os.mkdir(error_divide_list_save_dir)

    error_divide_img_save_dir = os.path.join(
            fail_config.fail_img_dir, 
            "error_data/error_img_from_normal_{}_divide".format(fail_config.model_type))
    if not os.path.exists(error_divide_img_save_dir):
        os.mkdir(error_divide_img_save_dir)


    for corrup in data_config.CORRUPTIONS:
        print("{}: ".format(corrup))

        error_list_path = os.path.join(error_list_dir, f'{corrup}.npy')
        error_list = np.load(error_list_path)

        img_list = [[] for _ in range(data_config.total_class_num)]
        for imgid in error_list:
            img_list[lbl_list[imgid]].append(imgid)
        
        train_list = []
        test_list = []
        for classid in range(data_config.total_class_num):
            # print(classid, len(np.random.choice(img_list[classid], sample_num_per_class)))
            train_list.append(np.random.choice(img_list[classid], size=fail_config.sample_num_per_class, replace=False))
        train_list = np.reshape(train_list, (-1))
        test_list = [ i for i in error_list if i not in train_list ]

        print("    save list: ", end='', flush=True)
        train_list_save_path = os.path.join(error_divide_list_save_dir, f"{corrup}_train.npy")
        test_list_save_path = os.path.join(error_divide_list_save_dir, f"{corrup}_test.npy")
        np.save(train_list_save_path, train_list)
        np.save(test_list_save_path, test_list)
        print("Done")

        print("    save img: ", end='', flush=True)
        cifar10_c_data_path = os.path.join(data_config.cifar10c_path, f'{corrup}.npy')
        cifar10_c_data_list = np.load(cifar10_c_data_path)
        error_divide_img_corrup_save_dir = os.path.join(error_divide_img_save_dir, corrup)
        if not os.path.exists(error_divide_img_corrup_save_dir):
            os.mkdir(error_divide_img_corrup_save_dir)
        for sets in ['train', 'test']:
            if sets == 'train':
                datalist = train_list
            elif sets == 'test':
                datalist = test_list
            for imgid in datalist:
                save_dir = os.path.join(error_divide_img_corrup_save_dir, sets)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                img = cifar10_c_data_list[imgid]
                save_path = os.path.join(save_dir, f"{imgid:0>5d}.jpg")
                cv2.imwrite(save_path, img)
        print("Done")


""" ---- cluster ---- """
def cluster_data(fail_config, data_config):
    lbl_file = os.path.join(data_config.cifar10c_path, "labels.npy")
    lbl_list = np.load(lbl_file)

    meta_path = os.path.join(
            fail_config.fail_img_dir, 
            "error_data/error_img_from_normal_{}_divide".format(fail_config.model_type))

    save_cluster_dir = os.path.join(
            fail_config.fail_img_dir, 
            "error_data/error_img_from_normal_{}_divide_cluster".format(fail_config.model_type))
    if not os.path.exists(save_cluster_dir):
        os.mkdir(save_cluster_dir)

    for corrup in data_config.CORRUPTIONS:
        # corrup = "brightness"
        print("-------- {} --------".format(corrup))

        img_data = [[], [], [], [], [], [], [], [], [], []]
        img_data_idx = [[], [], [], [], [], [], [], [], [], []]

        corrup_dir = os.path.join(meta_path, f'{corrup}/train/')
        img_list = os.listdir(corrup_dir)
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(corrup_dir, img_name)
            img = cv2.imread(img_path)

            img_idx = int(img_name[:-4])
            img_lbl = lbl_list[img_idx]

            img_data[img_lbl].append(img)
            img_data_idx[img_lbl].append(img_idx)

        class_idx = 0
        img_data_cluster = []
        for img_data_list in img_data:
            print("    class {}: ".format(class_idx), end='', flush=True)

            data = np.array(img_data_list)
            data = np.reshape(data, (len(data), -1))
            # print(np.shape(data))
            # print(np.shape(img_data_list))
            data = scale(data)
            kmeans = KMeans(init='k-means++', n_clusters=5, random_state=0)
            kmeans.fit(data)
            img_data_cluster.append(kmeans.predict(data))
            # print(img_data_cluster)

            print("Done")
            class_idx += 1
            # break

        img_info_list = []
        for classidx in range(len(img_data_idx)):
            for idx in range(len(img_data_idx[classidx])):
                img_info = (img_data_idx[classidx][idx], lbl_list[img_data_idx[classidx][idx]], img_data_cluster[classidx][idx])
                img_info_list.append(img_info)
                # break

        img_info_list.sort()
        # print(img_info_list)

        np.save(os.path.join(save_cluster_dir, f'{corrup}_train.npy'), img_info_list)
        # img_info_list = np.load(save_cluster_dir + corrup + '.npy')
        print("-------- Done --------")


""" ---- calculate distance ---- """
def calculate_dist(fail_config, data_config):
    cluster_path = os.path.join(
        fail_config.fail_img_dir,
        "error_data/error_img_from_normal_{}_divide_cluster".format(fail_config.model_type)
    )

    sample_path = os.path.join(
        fail_config.fail_img_dir,
        "error_data/error_img_from_normal_{}_divide_cluster_distance".format(fail_config.model_type)
    )
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    for corrup in data_config.CORRUPTIONS:
        print("{}: ".format(corrup), end='', flush=True)

        cluster_file = os.path.join(cluster_path, f'{corrup}_train.npy')

        info_data = np.load(cluster_file)
        # print(info_data)
        info_data_list = [[[] for i in range(5)] for j in range(10)]
        for info in info_data:
            imgid, classid, clusterid = info
            info_data_list[classid][clusterid].append(imgid)
        # print(info_data_list)

        cifar_img_file = os.path.join(data_config.cifar10c_path, f'{corrup}.npy')
        cifar10_c_data = np.load(cifar_img_file)
        imglist = np.reshape(cifar10_c_data, (len(cifar10_c_data), -1))

        res = []
        for classid in range(data_config.total_class_num):
            for clusterid in range(data_config.total_cluster_num):
                clu_avg = np.zeros(imglist[0].shape)
                for imgid in info_data_list[classid][clusterid]:
                    data = imglist[imgid]
                    clu_avg += data / len(info_data_list[classid][clusterid])
                for imgid in info_data_list[classid][clusterid]:
                    imgdata = imglist[imgid]
                    imgdist = (np.sum((imgdata - clu_avg) ** 2)) ** 0.5
                    imginfo = (imgid, classid, clusterid, imgdist)
                    res.append(imginfo)
        res.sort()
        res = np.array(res)
        np.save(os.path.join(sample_path, "cluster_info_{}.npy".format(corrup)), res)

                # imgid = np.random.choice(info_data_list[classid][clusterid])
                # sample_img = cifar10_c_data[imgid]
                # save_sample_img_path = save_sample_corrup_path + "{}_{}_{:0>5d}.jpg".format(classid, clusterid, imgid)
                # cv2.imwrite(save_sample_img_path, sample_img)

        print("Done")


""" ---- sampling reference image ---- """
def sampling_ref_img(fail_config, data_config):
    cluster_path = os.path.join(
        fail_config.fail_img_dir,
        "error_data/error_img_from_normal_{}_divide_cluster".format(fail_config.model_type)
    )

    sample_path = os.path.join(
        fail_config.fail_img_dir,
        "error_data/error_img_from_normal_{}_divide_cluster_sampling_img".format(fail_config.model_type)
    )
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    for corrup in data_config.CORRUPTIONS:
        print("{}: ".format(corrup), end='', flush=True)

        cluster_file = os.path.join(cluster_path, f'{corrup}_train.npy')

        info_data = np.load(cluster_file)
        # print(info_data)
        info_data_list = [[[] for i in range(5)] for j in range(10)]
        for info in info_data:
            imgid, classid, clusterid = info
            info_data_list[classid][clusterid].append(imgid)
        # print(info_data_list)

        save_sample_corrup_path = os.path.join(sample_path, f'{corrup}/train')
        if not os.path.exists(save_sample_corrup_path):
            os.makedirs(save_sample_corrup_path)

        cifar_img_file = os.path.join(data_config.cifar10c_path, f'{corrup}.npy')
        cifar10_c_data = np.load(cifar_img_file)
        for classid in range(data_config.total_class_num):
            for clusterid in range(data_config.total_cluster_num):
                imgid = np.random.choice(info_data_list[classid][clusterid])
                sample_img = cifar10_c_data[imgid]
                save_sample_img_path = os.path.join(save_sample_corrup_path, "{}_{}_{:0>5d}.jpg".format(classid, clusterid, imgid))
                cv2.imwrite(save_sample_img_path, sample_img)

        print("Done")


""" ---- collect data ---- """
def collect_data(fail_config, data_config):
    # ---- image ----
    cluster_dir = os.path.join(
        fail_config.fail_img_dir, 
        "error_data/error_img_from_normal_{}_divide_cluster_sampling_img/".format(fail_config.model_type)
    )

    whole_dir = os.path.join(cluster_dir, "whole/train/")
    if not os.path.exists(whole_dir):
        os.makedirs(whole_dir)

    for corrup in data_config.CORRUPTIONS:
        print("{} -- {}: ".format(fail_config.model_type, corrup), end='', flush=True)

        data_dir = os.path.join(cluster_dir + f"{corrup}/train/")

        os.system("cp {}* {}".format(data_dir, whole_dir))

        print("Done")

    # ---- npy ----
    base_c_failure_case_path = os.path.join(
        fail_config.fail_img_dir,
        "error_data/error_list_from_normal_{}_divide".format(fail_config.model_type)
    )

    whole_img = None
    whole_tag = None
    for corrup in data_config.CORRUPTIONS:
        id_failure_list = np.load(os.path.join(base_c_failure_case_path, f'{corrup}_test.npy'))
        img_failure_path = os.path.join(data_config.cifar10c_path, f"{corrup}.npy")
        img_failure_list = np.load(img_failure_path)
        img_failure_list = np.array(img_failure_list[id_failure_list])
        tag_failure_path = os.path.join(data_config.cifar10c_path, "labels.npy")
        tag_failure_list = np.load(tag_failure_path)
        tag_failure_list = np.array(tag_failure_list[id_failure_list])
        print("    -- {}: {}, {}".format(corrup, img_failure_list.shape, tag_failure_list.shape))

        if whole_img is None:
            whole_img = img_failure_list
            whole_tag = tag_failure_list
        else:
            whole_img = np.concatenate((whole_img, img_failure_list), axis=0)
            whole_tag = np.concatenate((whole_tag, tag_failure_list), axis=0)

    print("    -- {}: {}, {}".format("whole", whole_img.shape, whole_tag.shape))

    print("-- saving ... ", end='', flush=True)
    save_path = os.path.join(
        fail_config.fail_img_dir,
        "error_data/error_data_whole_set/from_{}/".format(fail_config.model_type)
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+"whole_data.npy", whole_img)
    np.save(save_path+"whole_label.npy", whole_tag)
    print("Done")








if __name__=="__main__":
    divide_train_test_data(fail_config, data_config)
    cluster_data(fail_config, data_config)
    calculate_dist(fail_config, data_config)
    sampling_ref_img(fail_config, data_config)
    collect_data(fail_config, data_config)
