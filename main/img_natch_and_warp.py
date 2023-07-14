import time

import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import glob
import scipy.io as sio

sys.path.append('..')
sys.path.append('../deep_feat_VGG16')
sys.path.append('../d2net')

sys.path.append('../superpoint')
import superpoint.demo_superpoint as superpoint

from rootsift import RootSIFT

import deep_feat_VGG16.DeepLKBatch as dlk
from deep_feat_VGG16.config import *

import d2net.d2net as d2net

import warnings
warnings.filterwarnings("ignore")


def scale_H(H, s):
    S = np.array([[s, 0, 0],[0, s, 0],[0, 0, 1]])
    H_scale = S @ H @ np.linalg.inv(S)
    return H_scale


def corner_loss(H_p, H_gt, patch_size, scale=1):
    # 给出Corner Loss函数
    H_p = scale_H(H_p, scale)
    if USE_CUDA:
        corners = Variable(torch.Tensor([[-patch_size/2, patch_size/2, patch_size/2, -patch_size/2],
                                [-patch_size/2, -patch_size/2, patch_size/2, patch_size/2],
                                [1, 1, 1, 1]]).double().cuda())
    else:
        corners = Variable(torch.Tensor([[-patch_size/2, patch_size/2, patch_size/2, -patch_size/2],
                                [-patch_size/2, -patch_size/2, patch_size/2, patch_size/2],
                                [1, 1, 1, 1]]).double())
    H_p = torch.tensor(H_p)
    H_gt = torch.tensor(H_gt)
    # if USE_CUDA:
    #     corners = Variable(torch.Tensor([[0, patch_size, 0, -patch_size],
    #                             [0, 0, patch_size, patch_size],
    #                             [1, 1, 1, 1]]).double().cuda())
    # else:
    #     corners = Variable(torch.Tensor([[0, patch_size, 0, -patch_size],
    #                             [0, 0, patch_size, patch_size],
    #                             [1, 1, 1, 1]]).double())

    corners_w_p = H_p.mm(corners)
    corners_w_gt = H_gt.mm(corners)


    loss = ((corners_w_p[0:2, :] - corners_w_gt[0:2, :]) ** 2).mean().item()
    R_loss = np.sqrt(loss)
    R_loss_rel = R_loss / patch_size

    return R_loss_rel


def test_dlk_single_match(img1, img2, idx=0, save_path=None):
    # warp_hmg里面会求逆，所以这里不用求逆

    scaled_im_height = 100  # 这样速度会快很多
    scaled_im_weight = round(img1.shape[1] * scaled_im_height / img1.shape[0])
    img1_PIL = Image.fromarray(img1)
    img1_PIL_rz = img1_PIL.resize((scaled_im_weight, scaled_im_height))
    img1_np_rz = transform.ToTensor()(img1_PIL_rz).numpy()

    img2_PIL = Image.fromarray(img2)
    img2_PIL_rz = img2_PIL.resize((scaled_im_weight, scaled_im_height))
    img2_np_rz = transform.ToTensor()(img2_PIL_rz).numpy()

    # 经过以上的转换，第一维已经变成了channel数
    if img1.shape[-1] == 3:
        img1_swap = np.swapaxes(img1, 0, 2)
        img1_swap = np.swapaxes(img1_swap, 1, 2)
    if img2.shape[-1] == 3:
        img2_swap = np.swapaxes(img2, 0, 2)
        img2_swap = np.swapaxes(img2_swap, 1, 2)

    img1_tens = Variable(torch.from_numpy(img1_np_rz).float()).unsqueeze(0)
    img1_tens_nmlz = dlk.normalize_img_batch(img1_tens)
    img2_tens = Variable(torch.from_numpy(img2_np_rz).float()).unsqueeze(0)
    img2_tens_nmlz = dlk.normalize_img_batch(img2_tens)

    dlk_net = dlk.DeepLK(dlk.custom_net(model_path))
    p_lk, _, itr_dlk = dlk_net(img1_tens_nmlz, img2_tens_nmlz, tol=1e-4, max_itr=max_itr_dlk, conv_flag=1, ret_itr=True)

    # 疑问：这里算出来的p_lk是不是能与SIFT算出来的P-lk等价
    p_lk_np = p_lk.cpu().squeeze().detach().numpy()
    p_lk_np = np.append(p_lk_np, 0)
    H_dlk_inv = np.reshape(p_lk_np, (3, 3)) + np.eye(3)
    H_dlk = np.linalg.inv(H_dlk_inv)
    Perspective_img = cv2.warpPerspective(img1, H_dlk, (img1.shape[1], img1.shape[0]))

    img1_rz = img1
    img1_rz_tens = transforms.ToTensor()(img1_rz)
    img1_rz_curr_tens = img1_rz_tens.float().unsqueeze(0)
    M_tmpl_w, _, xy_cor_curr_opt = dlk.warp_hmg(img1_rz_curr_tens, p_lk)
    M_tmpl_w_np = M_tmpl_w[0, :, :, :].cpu().detach().numpy()
    temp = np.swapaxes(M_tmpl_w_np, 0, 2)
    M_tmpl_w_np = np.swapaxes(temp, 0, 1)

    if save_path is None:
        cv2.imwrite('./dlk_warp_result/warped_{}.jpg'.format(idx), Perspective_img)
    else:
        cv2.imwrite(save_path.format('warp'), Perspective_img)

    return H_dlk


def test_sift_single_match(template, img, idx=0, RANSAC_para=3, save_path=None):
    h, w, _ = img.shape

    if template.shape[0] == 3:
        template = np.swapaxes(template, 0, 2)
        template = np.swapaxes(template, 0, 1)
        template = (template * 255).astype('uint8')

    if img.shape[0] == 3:
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        img = (img * 255).astype('uint8')

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    # sift = cv2.xfeatures2d.SURF_create()
    # sift = cv2.SIFT()

    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: (x.distance) * (1))
    img3 = cv2.drawMatches(template, kp1, img, kp2, matches[:30], None, flags=2)
    # plt.imshow(img3)
    # plt.show()
    if save_path is None:
        cv2.imwrite('./sift_match_result/match_{}.jpg'.format(idx), img3)
    else:
        cv2.imwrite(save_path.format('match'), img3)

    if (len(kp1) >= 10) and (len(kp2) >= 10):

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 这里的点对不做归一化可否？
        # src_pts = src_pts - w / 2
        # dst_pts = dst_pts - w / 2

        if (src_pts.size <= 5) or (dst_pts.size <= 5):
            H_found = np.eye(3)
        else:
            try:
                H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            except cv2.error:
                H_found = np.eye(3)
        # print(src_pts.shape)
        # print(dst_pts.shape)
        # H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if H_found is None:
            H_found = np.eye(3)

    else:
        H_found = np.eye(3)

    Perspective_img = cv2.warpPerspective(template, H_found, (template.shape[1], template.shape[0]))
    if save_path is None:
        cv2.imwrite('./sift_warp_result/match_{}.jpg'.format(idx), Perspective_img)
    else:
        cv2.imwrite(save_path.format('warp'), Perspective_img)

    return H_found


def test_d2net_single_match(img1_path, img2_path, idx=0, save_path=None):
    [image_1_keypoints, image_1_scores, image_1_descriptors] = d2net.d2net_extractor(img1_path)
    [image_2_keypoints, image_2_scores, image_2_descriptors] = d2net.d2net_extractor(img2_path)

    image_1 = cv2.imread(img1_path)
    image_2 = cv2.imread(img2_path)

    image_1_kpts = []

    for keypoint in image_1_keypoints:
        image_1_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))
    image_2_kpts = []

    for keypoint in image_2_keypoints:
        image_2_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    if (len(image_1_kpts) >= 10) and (len(image_2_kpts) >= 10):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Use D2-Net dep
        matches = flann.knnMatch(image_1_descriptors, image_2_descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.95 * n.distance:
                good.append(m)

        src_pts = np.float32([image_1_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_kpts = []
        for keypoint in src_pts:
            src_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))
        dst_pts = np.float32([image_2_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_kpts = []
        for keypoint in dst_pts:
            dst_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))

        if (src_pts.size <= 3) or (dst_pts.size <= 3):
            H_found = np.eye(3)
        else:
            H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        if H_found is None:
            H_found = np.eye(3)

    else:
        H_found = np.eye(3)


    Perspective_img = cv2.warpPerspective(image_1, H_found, (image_1.shape[1], image_1.shape[0]))
    if save_path is None:
        cv2.imwrite('./d2net_warp_result/match_{}.jpg'.format(idx), Perspective_img)
    else:
        cv2.imwrite(save_path.format('match'), Perspective_img)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(image_1_descriptors, image_2_descriptors)
    matches = sorted(matches, key=lambda x: (x.distance) * (1))
    image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
    img3 = cv2.drawMatches(image_1, image_1_kpts, image_2, image_2_kpts, matches[:10], None, flags=2)
    if save_path is None:
        cv2.imwrite('./d2net_match_result/match_{}.jpg'.format(idx), img3)
    else:
        cv2.imwrite(save_path.format('warp'), img3)
    return H_found


def test_superpoint_single_match(img1_path, img2_path, idx=0, save_path=None):
    interp = cv2.INTER_AREA
    weights_path = '../superpoint/superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    cuda = True
    fe = superpoint.SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda)

    # 待查询的图像路径
    image_1 = cv2.imread(img1_path, 0)
    image_1_color = cv2.imread(img1_path)
    image_1 = cv2.resize(image_1, (500, 500), interpolation=interp)
    image_1_color = cv2.resize(image_1_color, (500, 500), interpolation=interp)
    image_1 = (image_1.astype('float32') / 255.)
    img_h, img_w = image_1.shape
    pts1, image_1_descriptors, heatmap1 = fe.run(image_1)
    image_1_descriptors = np.swapaxes(image_1_descriptors, 0, 1)
    image_1_kpts = []
    pts1 = np.swapaxes(pts1, 0, 1)
    for keypoint in pts1:
        image_1_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    image_2 = cv2.imread(img2_path, 0)
    image_2_color = cv2.imread(img2_path)
    image_2 = cv2.resize(image_2, (500, 500), interpolation=interp)
    image_2_color = cv2.resize(image_2_color, (500, 500), interpolation=interp)
    image_2 = (image_2.astype('float32') / 255.)
    pts2, image_2_descriptors, heatmap2 = fe.run(image_2)
    time_check_2 = time.time()
    image_2_descriptors = np.swapaxes(image_2_descriptors, 0, 1)
    image_2_kpts = []
    pts2 = np.swapaxes(pts2, 0, 1)
    for keypoint in pts2:
        image_2_kpts.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], size=1))

    if (len(image_1_kpts) >= 10) and (len(image_2_kpts) >= 10):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 使用k近邻匹配的方式，所以这里的返回值是2，
        matches = flann.knnMatch(image_1_descriptors, image_2_descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) > 10:
            src_pts = np.float32([image_1_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            src_kpts = []
            for keypoint in src_pts:
                src_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))
            dst_pts = np.float32([image_2_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_kpts = []
            for keypoint in dst_pts:
                dst_kpts.append(cv2.KeyPoint(x=keypoint[0, 0], y=keypoint[0, 1], size=1))

            if (src_pts.size <= 3) or (dst_pts.size <= 3):
                H_found = np.eye(3)
            else:
                H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                # print(H_found.mean())
                # print(H_found)
        else:
            H_found = np.eye(3)
    else:
        H_found = np.eye(3)

    Perspective_img = cv2.warpPerspective(image_1_color, H_found, (image_1.shape[1], image_1.shape[0]))
    if save_path is None:
        cv2.imwrite('./spp_warp_result/warp_{}.jpg'.format(idx), Perspective_img)
    else:
        cv2.imwrite(save_path.format('warp'), Perspective_img)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(image_1_descriptors, image_2_descriptors)
    matches = sorted(matches, key=lambda x: (x.distance) * (1))
    # image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    # image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
    img3 = cv2.drawMatches(image_1_color.astype('uint8'), image_1_kpts,
                           image_2_color.astype('uint8'), image_2_kpts, matches[:30], None, flags=2)
    if save_path is None:
        cv2.imwrite('./spp_match_result/match_{}.jpg'.format(idx), img3)
    else:
        cv2.imwrite(save_path.format('match'), img3)

    return H_found


def test_Root_sift_single_match(template, img, idx=0, RANSAC_para=3, save_path=None):
    h, w, _ = img.shape

    if template.shape[0] == 3:
        template = np.swapaxes(template, 0, 2)
        template = np.swapaxes(template, 0, 1)
        template = (template * 255).astype('uint8')

    if img.shape[0] == 3:
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        img = (img * 255).astype('uint8')

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rs = RootSIFT.RootSIFT()
    kp1, des1 = rs.compute(template_gray)
    kp2, des2 = rs.compute(img_gray)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: (x.distance) * (1))
    img3 = cv2.drawMatches(template, kp1, img, kp2, matches[:30], None, flags=2)
    if save_path is None:
        cv2.imwrite('./root_sift_match_result/match_root_{}.jpg'.format(idx), img3)
    else:
        cv2.imwrite(save_path.format('match'), img3)

    if (len(kp1) >= 10) and (len(kp2) >= 10):

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 是否需要对于这里的对点坐标进行归一化操作
        # src_pts = src_pts - w / 2
        # dst_pts = dst_pts - w / 2

        if (src_pts.size <= 5) or (dst_pts.size <= 5):
            H_found = np.eye(3)
        else:
            try:
                H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            except cv2.error:
                H_found = np.eye(3)

        if H_found is None:
            H_found = np.eye(3)

    else:
        H_found = np.eye(3)

    Perspective_img = cv2.warpPerspective(template, H_found, (template.shape[1], template.shape[0]))
    if save_path is None:
        cv2.imwrite('./root_sift_warp_result/warped_root_{}.jpg'.format(idx), Perspective_img)
    else:
        cv2.imwrite(save_path.format('warp'), Perspective_img)

    return H_found


def dataset_test_img_patch_warp():
    dlk_loss_list = []
    sift_loss_list = []
    d2net_loss_list = []
    spp_loss_list = []
    images_dir = '../satellite_imagery_dataset/farmland_image_match/warp_patch_edge/*.png'
    image_path_list = glob.glob(images_dir)
    image_path_list = sorted(image_path_list)
    """
    for i in image_path_list:
        print(i+'\n')
    """
    print("The number of images is ", len(image_path_list))
    for patch_idx in range(6):
        for i in range(100):
            image_path_samp = image_path_list[i*2+patch_idx*50]

            # 把第二张变成第一张的样子，但是程序中一般都是把参数一变成参数二，所以这里顺序比较奇怪
            img1_path = image_path_list[i*2+patch_idx*50+1]
            img2_path = image_path_list[i*2+patch_idx*50]
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            h, w, _ = img1.shape

            # DLK 方法
            dlk_time_record_1 = time.time()
            H_dlk = test_dlk_single_match(img1, img2, i + patch_idx * 50)
            dlk_time_record_2 = time.time()

            # SIFT 方法
            sift_time_record_1 = time.time()
            H_sift = test_sift_single_match(img1, img2, i + patch_idx * 50)
            sift_time_record_2 = time.time()

            # D2Net 方法
            d2net_time_record_1 = time.time()
            # H_d2net = test_d2net_single_match(img1_path, img2_path, i + patch_idx * 50)
            H_d2net = np.eye(3)
            d2net_time_record_2 = time.time()

            # Superpoint 方法
            spp_time_record_1 = time.time()
            H_spp = test_superpoint_single_match(img1_path, img2_path, i + patch_idx * 50)
            spp_time_record_2 = time.time()

            warp_p_list = np.load('../satellite_imagery_dataset/huangmo_image_match/warp_para/warp_p_{}.npy'.format(patch_idx))
            path_idx = int((image_path_samp.split('_'))[-2])
            H_gt_inv = np.reshape(warp_p_list[9 + 9 * path_idx:18 + 9 * path_idx], (3, 3)) + np.eye(3)
            H_gt = np.linalg.inv(H_gt_inv)


            dlk_loss = corner_loss(H_dlk, H_gt, h, h/100)
            dlk_loss_list.append(dlk_loss)

            sift_loss = corner_loss(H_sift, H_gt, h)
            sift_loss_list.append(sift_loss)

            d2net_loss = corner_loss(H_d2net, H_gt, h)
            d2net_loss_list.append(d2net_loss)

            spp_loss = corner_loss(H_spp, H_gt, h)
            spp_loss_list.append(spp_loss)

            print("DLK Corner Loss = ", dlk_loss)
            print("sift Corner Loss = ", sift_loss)
            print("d2net Corner Loss = ", d2net_loss)
            print("superpoint Corner Loss = ", spp_loss)

            print("DLK spend time = ", dlk_time_record_2 - dlk_time_record_1)
            print("sift spend time = ", sift_time_record_2 - sift_time_record_1)
            print("d2net spend time = ", d2net_time_record_2 - d2net_time_record_1)
            print("superpoint spend time = ", spp_time_record_2 - spp_time_record_1)

    np.save("../satellite_imagery_dataset/farmland_image_match/dlk_loss.npy", dlk_loss_list)
    np.save("../satellite_imagery_dataset/farmland_image_match/sift_loss.npy", sift_loss_list)
    np.save("../satellite_imagery_dataset/farmland_image_match/d2net_loss.npy", d2net_loss_list)
    np.save("../satellite_imagery_dataset/farmland_image_match/spp_loss.npy", spp_loss_list)


def single_test_img_patch_warp():
    path_to_save = 'E:\\Datasets\\VLD\\SampleImages\\University1652\\sample1\\'
    img_path_list = glob.glob('E:\\Datasets\\VLD\\SampleImages\\University1652\\sample1\\*')
    img1_fname, img2_fname = img_path_list[0], img_path_list[1]
    img1 = cv2.imread(img1_fname)
    img2 = cv2.imread(img2_fname)

    t1 = time.time()
    H_dlk = test_dlk_single_match(img1, img2, save_path=path_to_save+'dlk_{}.jpg')
    t2 = time.time()
    print("dlk methods spends time: {:2.3}".format(t2 - t1))
    print("dlk methods homography matrix: \n", H_dlk)
    print('\n')

    t1 = time.time()
    H_sift_inv = test_sift_single_match(img1, img2, save_path=path_to_save+'sift_{}.jpg')
    t2 = time.time()
    print("sift methods spends time: {:2.3}".format(t2 - t1))
    print("sift methods homography matrix: \n", H_sift_inv)
    print('\n')

    t1 = time.time()
    H_rootsift_inv = test_Root_sift_single_match(img1, img2, save_path=path_to_save+'root_sift_{}.jpg')
    t2 = time.time()
    print("rootsift methods spends time: {:2.3}".format(t2 - t1))
    print("rootsift methods homography matrix: \n", H_rootsift_inv)
    print('\n')

    t1 = time.time()
    H_spp = test_superpoint_single_match(img1_fname, img2_fname, 0, save_path=path_to_save+'spp_{}.jpg')
    t2 = time.time()
    print("spp methods spends time: {:2.3}".format(t2 - t1))
    print("spp methods homography matrix: \n", H_spp)
    print('\n')

    t1 = time.time()
    H_spp = test_d2net_single_match(img1_fname, img2_fname, 0, save_path=path_to_save+'d2net_{}.jpg')
    t2 = time.time()
    print("d2net methods spends time: {:2.3}".format(t2 - t1))
    print("d2net methods homography matrix: \n", H_spp)
    print('\n')


def main():
    # dataset_test_img_patch_warp()
    # 对于匹配对验证的工作需要增加使用，修改代码的模块化程度
    single_test_img_patch_warp()


if __name__ == "__main__":
    main()
