python3 ./train.py \
--img 640 --batch 8 --epochs 25 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/warmup_epoch/hyp.resume.run27.yaml

python3 ./train.py \
--img 640 --batch 8 --epochs 25 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/warmup_epoch/hyp.resume.run28.yaml

python3 ./train.py \
--img 640 --batch 8 --epochs 25 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/warmup_epoch/hyp.resume.run29.yaml

python3 ./train.py \
--img 640 --batch 8 --epochs 25 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/warmup_epoch/hyp.resume.run30.yaml

python3 ./train.py \
--img 640 --batch 8 --epochs 25 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/lr/hyp.resume.run31.yaml

python3 ./train.py \
--img 640 --batch 8 --epochs 25 --workers 16 \
--data ./data/open_images_filtered_v3.yaml --cfg ./models/yolov5m.yaml \
--weights ./yolov5m.pt \
--hyp ./data/hyp/lr/hyp.resume.run32.yaml