python3 ./train.py \
--img 640 --batch 8 --epochs 40 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/hyp.resume.run23_ud.yaml