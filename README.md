# yolact-test

A test version of Yolact in PyTorch for instance segmentation

## Acknowledgement
 - This repository references [dbolya](https://github.com/dbolya/yolact)'s work.

## Dataset
 - Check COCO 2017 dataset
   ```
   python dataset_player.py
   python dataset_player.py --training
   ```
 - Check KITTI dataset
   ```
   python dataset_player.py --dataset=kitti_dataset
   python dataset_player.py --dataset=kitti_dataset --training
   ```
 - Check SEUMM HQ dataset
   ```
   python dataset_player.py --dataset=seumm_hq_dataset
   python dataset_player.py --dataset=seumm_hq_dataset --training
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
 - Train on SEUMM HQ dataset
   ```
   python train.py --config=yolact_resnet50_config --dataset=seumm_hq_dataset
   ```

## Evaluation
 - Evaluate on COCO 2017 dataset
   ```
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --display
   ```
 - Evaluate on KITTI dataset
   ```
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth --display
   ```
 - Evaluate on SEUMM HQ dataset
   ```
   # NotImplementError
   ```
 - The result should be

| Backbone | Dataset  | Iter | val mAP@.5 B | val mAP@.5:.95 B | val mAP@.5 M | val mAP@.5:.95 M |
|:--------:|:--------:|:----:|:------------:|:----------------:|:------------:|:----------------:|
| ResNet50 | COCO     | 380k | 46.56        | 27.35            | 42.75        | 25.78            |
| ResNet50 | KITTI    | 60k  | 44.67        | 24.23            | 39.55        | 22.34            |
| ResNet50 | SEUMM HQ | 60k  |              |                  |              |                  |

## Demo
 - Run a demo with COCO 2017 model
   ```
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --image=my_image.jpeg --score_threshold=0.25 --top_k=20
   python eval.py --trained_model=weights/coco/yolact_resnet50_25_380000.pth --images=test_coco:test_out_coco --score_threshold=0.25 --top_k=20
   ```
 - Run a demo with KITTI model
   ```
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth --image=my_image.jpeg --score_threshold=0.25 --top_k=20
   python eval.py --dataset=kitti_dataset --trained_model=weights/kitti/yolact_resnet50_107_60000.pth --images=test_kitti:test_out_kitti --score_threshold=0.25 --top_k=20
   ```
 - Run a demo with SEUMM HQ model
   ```
   # NotImplementError
   ```
   

