python3 ../detect_seg_rot.py \
--weights /home/osense-office/Documents/ext_repo/yolov5/runs/exp23/weights/best.pt \
--source_type video \
--img-size 3000 \
--source '/home/osense-office/Desktop/camera/213/ffmpeg_213_2020-10-15_15-10-00.mp4' \
--output /home/osense-office/Desktop/camera/213/test \
--device 0 \
--classes 0 \
--conf-thres 0.5 \
--iou-thres 0.5 \
--delay 20 \
--stream_scale 0.4 \
--rotate 0 \
--view-bbox \
--save-bbox \
--background '/home/osense-office/Desktop/fisheye/217/background_late.jpg' \
--seg_config '/home/osense-office/Documents/ext_repo/yolov5/data/segment/horizontal_split.yaml'