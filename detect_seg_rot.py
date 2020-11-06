import argparse
import os
import numpy as np
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from pathlib import Path
import pickle

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadVideo, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, rotate_bbox)
from utils.density_map_utils import (
    plot_one_density_distribution, setup_density_map
)
from utils.output_utils import get_new_video_writer
from utils.stream_utils import stream_result
from utils.preprocess_utils import get_foreground_mask
from utils.torch_utils import select_device, load_classifier, time_synchronized

vid_path = None
vid_writer = None
INV_255 = 1. / 255.

def detect(write_label=False):
    # basic options
    out, source_type, source, weights, imgsz = \
        opt.output, opt.source_type, opt.source, opt.weights, opt.img_size,

    # options to stream results
    view_bbox, view_dmap = \
        opt.view_bbox, opt.view_dmap

    # options to save results
    save_txt, save_bbox, save_dmap, save_pickle, save_frame = \
        opt.save_txt, opt.save_bbox, opt.save_dmap, opt.save_pickle, opt.save_frame

    # options to segment the input frame
    seg_config = opt.seg_config

    webcam = None
    if source_type == 'webcam':
        webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # TODO: replace with Path as in pathlib
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer, dmap_vid_writer = None, None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, seg_config, img_size=imgsz)
    elif source_type == 'images':
        save_img = True
        dataset = LoadImages(source, seg_config, img_size=imgsz)
    else:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadVideo(source, seg_config, img_size=imgsz)
        vid_path = opt.source

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    if save_pickle:
        print('save pickle')
        pickle_dict = {}

    frame_count = 0

    for path, imgs, im0s, vid_cap in dataset:
        # Here, im0s is the original image loaded using OpenCV
        frame_count += 1
        print("length of subimages %d" % len(imgs))
        """
        for img in imgs:
            img = cv2.resize(img, None, fx=0.2, fy=0.2)
            cv2.imshow("test", img)
            cv2.waitKey(0)
        """
        imgs = [torch.from_numpy(img).to(device) for img in imgs]
        imgs = [img.half() if half else img.float() for img in imgs]  # uint8 to fp16/32
        imgs = [img * INV_255 for img in imgs]  # 0 - 255 to 0.0 - 1.0

        for img in imgs:
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            if save_dmap or view_dmap:
                dmap_rotate, pos_rotate = setup_density_map(img)
                dmap_vid_path, dmap_vid_writer = None, None

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s_rotate)

            file_name = Path(path).name
            if save_pickle:
                pickle_dict[file_name] = []

            if save_frame:  # save raw frame
                frame_path = str(Path(out) / file_name / f'{str(frame_count)}.jpg')
                cv2.imwrite(frame_path, im0s)

        # Process detections
        for i, detection in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s_rotate[i]
                orig_im0 = np.rot90(im0s_rotate[i], k=-opt.rotate, axes=(0, 1)).copy()
            else:
                p, s, im0 = path, '', im0s_rotate
                orig_im0 = np.rot90(im0s_rotate, k=-opt.rotate, axes=(0, 1)).copy()

            # variables for density map
            dmap_file_name = f'dmap_{Path(p).name}'
            dmap_save_path = str(Path(out) / dmap_file_name)  # change the name
            dmap_vid_path = None
            save_path = str(Path(out) / Path(p).name)

            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'image' else '')
            s += '%gx%g ' % img_rotate.shape[2:]  # print string
            gn = torch.tensor(im0s_rotate.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if detection is not None and len(detection):
                # Rescale boxes from img_size to im0 size
                detection[:, :4] = scale_coords(img_rotate.shape[2:], detection[:, :4], im0.shape).round()

                # Print results
                for c in detection[:, -1].unique():
                    n = (detection[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(detection):
                    xyxy_unrotate = rotate_bbox(im0.shape, torch.tensor(xyxy).view(1, 4), k=-opt.rotate).squeeze(0)
                    xywh_unrotate = xyxy2xywh(torch.tensor(xyxy_unrotate).view(1, 4))

                    if save_txt:  # Write to file
                        xywh = (xywh_unrotate / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_pickle:
                        xywhc = xywh_unrotate.numpy().tolist() #TODO: there are still negative x!?? what the heck
                        xywhc.append(conf.data.cpu().item())
                        pickle_dict[file_name].append(xywhc)

                    if save_bbox or view_bbox:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, write_label=write_label, color=colors[int(cls)],
                                     line_thickness=1)
                        plot_one_box(xyxy_unrotate, orig_im0, label=label, write_label=write_label, color=colors[int(cls)],
                                     line_thickness=1)

                    if save_dmap or view_dmap:
                        dmap_rotate = plot_one_density_distribution(xyxy, pos_rotate, dmap_rotate)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_bbox or view_dmap:
                if view_dmap:
                    dmap_rotate *= 255 * 40
                    dmap_rotate[dmap_rotate > 255] = 255
                    # plt.imshow(dmap_rotate)
                    # plt.show()
                    im0[:, :, 2] = dmap_rotate[:, :, 0]
                stream_result(im0, scale=opt.stream_scale, mode=dataset.mode, window_name=p)

            # Save results (image with detections). We always rotate the image back to its original orientation
            if save_bbox or save_dmap:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, orig_im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        vid_writer = get_new_video_writer(save_path, vid_writer, vid_cap) # new video
                    vid_writer.write(im0)

            if save_dmap:
                dmap_rotate = np.rot90(dmap_rotate, k=-opt.rotate)
                if dataset.mode == 'images':
                    cv2.imwrite(dmap_save_path, dmap_rotate)
                else:
                    if dmap_vid_path != dmap_save_path:
                        dmap_vid_writer = get_new_video_writer(dmap_save_path, dmap_vid_writer, vid_cap)  # new video
                    dmap_vid_writer.write(dmap_rotate)


    if save_txt or save_img:
        print('results saved to %s' % Path(out))

    if save_pickle:
        with open(Path(out) / f"thres{opt.conf_thres}.pickle", 'wb') as handle:
            pickle.dump(pickle_dict, handle)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source_type', type=str, default='image', help='the type of source [video | webcam | image]')

    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder

    parser.add_argument('--seg_config', type=str, default='data/segment/segment_trivial', help='Segmentation parameter path; if None, \
                            then no segmentation will be performed')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--background', type=str, default=None, help='The path to the mask image that masked out most of the background')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # options to stream the results
    parser.add_argument('--view-bbox', action='store_true', help='display detection results with drawn bounding box')
    parser.add_argument('--view-dmap', action='store_true', help='display detection results with density map')
    parser.add_argument('--stream_scale', type=float, default=1.0, help='the width of the shown image or video')
    parser.add_argument('--delay', type=int, default=1, help='The delay when displaying using OpenCV')

    #
    parser.add_argument('--rotate', type=int, default=0, help='rotate the frame counter clock wise n*90 degrees')
    parser.add_argument('--with-label', action='store_true', help='Set to false to hide the labels in the result image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')

    # options regarding writing the result to disk
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-bbox', action='store_true', help='save original image with bbox drawn in red')
    parser.add_argument('--save-dmap', action='store_true', help='Whether to save the density map')
    parser.add_argument('--save-pickle', action='store_true', help='Whether to save the detection result to a pickle file')
    parser.add_argument('--save-frame', action='store_true', help='Save frame as .jpg file in video or webcam mode')
    parser.add_argument('--save-rotate', action='store_true', help='Change the orientation of the saved image')
    parser.add_argument('--max-frame', action='store_true', help='Maximum number of frames to process in video or webcam mode')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(write_label=opt.with_label)
