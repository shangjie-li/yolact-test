# yolact-test

A test version of Yolact in PyTorch for instance segmentation

## Acknowledgement
 - This repository references [dbolya](https://github.com/dbolya/yolact)'s work.

## Dataset
 - Check the COCO 2017 dataset
   ```
   python dataset_player.py
   ```
 - Check the KITTI dataset
   ```
   python dataset_player.py kitti
   ```

## Training
 - Train on COCO 2017 dataset
   ```
   python train.py --config=yolact_resnet50_config
   ```
 - Train on KITTI dataset
   ```
   python train.py --config=yolact_resnet50_config --dataset=kitti
   ```
   
## Evaluation
 - Evaluate on COCO 2017 dataset (mAP: 20.13 for box & 19.77 for mask)
   ```
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --display
   ```
 - Evaluate on KITTI dataset (mAP: 22.08 for box & 20.77 for mask)
   ```
   python eval.py --dataset=kitti --trained_model=weights/kitti/yolact_base_89_50000.pth
   python eval.py --dataset=kitti --trained_model=weights/kitti/yolact_base_89_50000.pth --display
   ```
   
## Demo
 - Run on the image with COCO 2017 model
   ```
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --image=my_image.jpeg --score_threshold=0.5 --top_k=20
   ```
 - Run on the image with KITTI model
   ```
   python eval.py --dataset=kitti --trained_model=weights/kitti/yolact_base_89_50000.pth --image=my_image.jpeg --score_threshold=0.5 --top_k=20
   ```
   

