import torch 
import cv2
import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation
import scipy

# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization

# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k0 = np.clip(k0, 0, colorwheel.shape[0]-1)
    k1 = k0 + 1
    k1 = np.clip(k1, 0, colorwheel.shape[0]-1)
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

# MIT License
#
# Copyright (c) 2023 Alex Spirin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Alex Spirin
# Date Created: 2023-08-24

def edge_detector(image, threshold=0.5, edge_width=1):
    """
    Detect edges in an image with adjustable edge width.

    Parameters:
        image (numpy.ndarray): The input image.
        edge_width (int): The width of the edges to detect.

    Returns:
        numpy.ndarray: The edge image.
    """
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Sobel edge map.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=edge_width)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=edge_width)

    # Compute the edge magnitude.
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize the magnitude to the range [0, 1].
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # Threshold the magnitude to create a binary edge image.

    edge_image = (mag > threshold).astype(np.uint8) * 255

    return edge_image

def get_unreliable(flow):
    # Mask pixels that have no source and will be taken from frame1, to remove trails and ghosting.

    # Calculate the coordinates of pixels in the new frame
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = x + flow[..., 0]
    new_y = y + flow[..., 1]

    # Create a mask for the valid pixels in the new frame
    mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)

    # Create the new frame by interpolating the pixel values using the calculated coordinates
    new_frame = np.zeros((flow.shape[0], flow.shape[1], 3))*1.-1
    new_frame[new_y[mask].astype(np.int32), new_x[mask].astype(np.int32)] = 255

    # Keep masked area, discard the image.
    new_frame = new_frame==-1
    return new_frame, mask



def remove_small_holes(mask, min_size=50):
    # Copy the input binary mask
    result = mask.copy()

    # Find contours of connected components in the binary image
    contours, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for i in range(len(contours)):
        # Compute the area of the i-th contour
        area = cv2.contourArea(contours[i])

        # Check if the area of the i-th contour is smaller than min_size
        if area < min_size:
            # Draw a filled contour over the i-th contour region
            cv2.drawContours(result, [contours[i]], 0, 255, -1, cv2.LINE_AA, hierarchy, 0)

    return result


def filter_unreliable(mask, dilation=1):
  img = 255-remove_small_holes((1-mask[...,0].astype('uint8'))*255, 200)
  img = binary_erosion(img, disk(1))
  img = binary_dilation(img, disk(dilation))
  return img

def get_flow_and_mask(frame1, frame2, num_flow_updates=20, raft_model=None, edge_width=11, dilation=2):

    with torch.autocast('cuda', dtype=torch.float16):
        raft_model.cuda()
        frame1 = frame1.transpose(1,-1).cuda()
        frame2 = frame2.transpose(1,-1).cuda()
        flow21 = raft_model(frame2.half(), frame1.half(), num_flow_updates=num_flow_updates)[-1] #flow_bwd
        mag = (flow21[:,0:1,...]**2 + flow21[:,1:,...]**2).sqrt()
        mag_thresh = 0.5
        #zero out flow values for non-moving frames below threshold to avoid noisy flow/cc maps
        if mag.max()<mag_thresh:
            flow21_clamped = torch.where(mag<mag_thresh, 0, flow21)
        else:
            flow21_clamped = flow21
        flow21 = flow21[0].permute(1, 2, 0).detach().cpu()
        flow21_clamped = flow21_clamped[0].permute(1, 2, 0).detach().cpu().numpy()

        flow12 = raft_model(frame1, frame2)[-1]
        flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

        predicted_flows_bwd = flow21_clamped
        predicted_flows = flow12

        flow_imgs = flow_to_image(predicted_flows_bwd)
        edge = edge_detector(flow_imgs.astype('uint8'), threshold=0.1, edge_width=edge_width)
        occlusion_mask, _ = get_unreliable(predicted_flows)
        _, overshoot = get_unreliable(predicted_flows_bwd)

        occlusion_mask = (torch.from_numpy(255-(filter_unreliable(occlusion_mask, dilation)*255)).transpose(0,1)/255).cpu()
        border_mask = (torch.from_numpy(overshoot*255).transpose(0,1)/255).cpu()
        edge_mask = (torch.from_numpy(255-edge).transpose(0,1)/255).cpu()
        print(flow_imgs.max(), flow_imgs.min())
        flow_imgs = (torch.from_numpy(flow_imgs.transpose(1,0,2))/255).cpu()[None,]
        raft_model.cpu()

    return flow21, flow_imgs, edge_mask, occlusion_mask, border_mask



def warp_flow(img, flow, mul=1.):
      h, w = flow.shape[:2]
      flow = flow.copy()
      flow[:, :, 0] += np.arange(w)
      flow[:, :, 1] += np.arange(h)[:, np.newaxis]
      flow*=mul
      res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)

      return res

def apply_warp(current_frame, flow, padding=0):
    pad_pct = padding
    flow21 = flow 
    current_frame = current_frame[0]
    if pad_pct>0:
        pad = int(max(flow21.shape)*pad_pct)
    print(current_frame.shape, flow21.shape)
    flow21 = np.pad(flow21.numpy(), pad_width=((pad,pad),(pad,pad),(0,0)),mode='constant')
    current_frame = np.pad(current_frame.numpy().transpose(1,0,2), pad_width=((pad,pad),(pad,pad),(0,0)),mode='reflect')
    print(flow21.max(), flow21.shape, flow21.dtype)
    warped_frame = warp_flow(current_frame , flow21).transpose(1,0,2)
    warped_frame = warped_frame[pad:warped_frame.shape[0]-pad,pad:warped_frame.shape[1]-pad,:]
    warped_frame = torch.from_numpy(warped_frame).cpu()

    return warped_frame[None, ]
  
def mix_cc(missed_cc, overshoot_cc, edge_cc, blur=2, dilate=0, missed_consistency_weight=1, 
           overshoot_consistency_weight=1, edges_consistency_weight=1, force_binary=True):
    #accepts 3 maps [h x w] 0-1 range 
    missed_cc = np.array(missed_cc)
    overshoot_cc = np.array(overshoot_cc)
    edge_cc = np.array(edge_cc)
    weights = np.ones_like(missed_cc)
    weights*=missed_cc.clip(1-missed_consistency_weight,1)
    weights*=overshoot_cc.clip(1-overshoot_consistency_weight,1)
    weights*=edge_cc.clip(1-edges_consistency_weight,1)
    if force_binary:
      weights = np.where(weights<0.5, 0, 1)
    if dilate>0:
      weights = (1-binary_dilation(1-weights, disk(dilate))).astype('uint8')
    if blur>0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])

    return torch.from_numpy(weights)