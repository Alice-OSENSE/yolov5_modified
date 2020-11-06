python3 ../detect.py \
--weights /home/osense-office/Documents/ext_repo/yolov5/runs/exp7/weights/best.pt \
--source /home/osense-office/Documents/dataset/single_img/dog_test.jpg \
--output /home/osense-office/Documents/ext_repo/yolov5/results \
--mode images \
--device 0 \
--view-img \
--classes 0 \
--conf-thres 0.35
