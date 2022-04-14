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
   python dataset_player.py kitti_dataset
   ```

## Training
 - Train on COCO 2017 dataset
   ```
   python train.py --config=yolact_resnet50_config
   ```
 - Train on KITTI dataset
   ```
   python train.py --config=yolact_resnet50_config --dataset=kitti_dataset
   ```
   
## Evaluation
 - Evaluate on COCO 2017 dataset (mAP: 20.13 for box & 19.77 for mask)
   ```
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --display
   ```
 - Evaluate on KITTI dataset (mAP: 22.08 for box & 20.77 for mask)
   ```
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth --display
   ```
 - The result should be
 
| Model           | Dataset | Iter | val mAP@.5(B) | val mAP@.5:.95(B) | val mAP@.5(M) | val mAP@.5:.95(M) |
|:---------------:|:-------:|:----:|:-------------:|:-----------------:|:-------------:|:-----------------:|
| yolact_resnet50 | COCO    | 380k | 46.56         | 27.35             | 42.75         | 25.78             |
| yolact_resnet50 | KITTI   | 60k  | 44.67         | 24.23             | 39.55         | 22.34             |

## Demo
 - Run on the image with COCO 2017 model
   ```
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --image=my_image.jpeg --score_threshold=0.5 --top_k=20
   ```
 - Run on the image with KITTI model
   ```
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth --image=my_image.jpeg --score_threshold=0.5 --top_k=20
   ```
   

