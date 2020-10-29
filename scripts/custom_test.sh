python3 ../custom_test.py \
--weights ../runs/exp24/weights/best.pt \
--data ../data/open_images_full.yaml \
--batch-size 1 \
--conf-thres 0.4 \
--iou-thres 0.65 \
--task test \
--device 0 \
--single-cls \
--verbose