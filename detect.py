import argparse
import os
import numpy as np
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadVideo
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def convert_rotatecode(rotate):
    if rotate == 1:
        return cv2.ROTATE_90_CLOCKWISE
    if rotate == 2:
        return cv2.ROTATE_180
    if rotate == 3:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None

def detect(save_img=False, write_label=False):
    out, source_type, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source_type, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = None
    if source_type == 'webcam':
        webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
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
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    elif source_type == 'image':
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, rotate=opt.rotate)
    else:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadVideo(source, img_size=imgsz, rotate=opt.rotate)
        vid_path = opt.source

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        im0s_rotate = np.rot90(im0s, k=opt.rotate, axes=(0, 1)).copy()
        img_rotate = np.rot90(img, k=opt.rotate, axes=(1, 2)).copy()
        img_rotate = torch.from_numpy(img_rotate).to(device)
        img_rotate = img_rotate.half() if half else img_rotate.float()  # uint8 to fp16/32
        img_rotate /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_rotate.ndimension() == 3:
            img_rotate = img_rotate.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img_rotate, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img_rotate, im0s_rotate)

        # Process detections
        for i, detection in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s_rotate[i].copy()
            else:
                p, s, im0 = path, '', im0s_rotate

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
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, write_label=write_label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                im0_resized = cv2.resize(im0, None, fx=opt.stream_scale, fy=opt.stream_scale)
                cv2.imshow(p, im0_resized)
                if dataset.mode == 'video':
                    cv2.waitKey(delay=1)
                elif cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)

                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        if opt.rotate % 2:
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        print(w)
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('sults saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source_type', type=str, default='image', help='the type of source [video | webcam | image]')
    parser.add_argument('--segment', type=str, default='data/segment/segment_trivial', help='Segmentation parameter path; if None, \
                        then no segmentation will be performed')
    # the path to video, if source_type == video
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--with-label', action='store_true', help='Set to false to hide the labels in the result image')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--delay', type=int, default=1, help='The delay when displaying usinOpenCV')
    parser.add_argument('--stream_scale', type=float, default=1.0, help='the width of the shown image or video')
    parser.add_argument('--rotate', type=int, default=0, help='rotate the frame counter clock wise n*90 degrees')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(save_img=True, write_label=opt.with_label)
