python3 ../detect.py \
--weights /home/osense-office/Documents/ext_repo/yolov5/runs/exp21/weights/best.pt \
--source_type video \
--source /home/osense-office/Desktop/ffmpeg_211_2020-08-29_11-40-00.mp4 \
--output /home/osense-office/Documents/ext_repo/yolov5/results \
--device 0 \
--view-img \
--classes 0 \
--conf-thres 0.45 \
--iou-thres 0.6 \
--delay 20 \
--stream_scale 0.4 \
--rotate 2