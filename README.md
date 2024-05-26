

<div align="center">

<samp>

<h2> Surgical-VQLA++: Adversarial Contrastive Learning for Calibrated Robust VQLA in Robotic Surgery </h1>

<h4> Long Bai*, Guankun Wang*, Mobarakol Islam*, Lalithkumar Seenivasan, An Wang and Hongliang Ren </h3>



</samp>   


---

</div>     


## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQLA/` : Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                    - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
        - `....`
        - `seq_16`
    - `EndoVis-17-VQLA/` : 97 frames are selected from EndoVIS17 challange for external validation. 
        - `left_frames`
        - `vqla`
            - `label`: Q&A pairs and bounding box label.
            - `img_features`: Contains img_features extracted from each frame with different patch size.
                - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
- `models/`: 
    - CATViLEmbedding.py : our proposed model for VQLA task.
    - DeiTPrediction.py ：DeiT encoder-based model for VQLA task.
    - VisualBertResMLP.py : VisualBERT ResMLP encoder from Surgical-VQA.
    - visualBertPrediction.py : VisualBert encoder-based model for VQLA task.
    - VisualBertResMLPPrediction.py : VisualBert ResMLP encoder-based model for VQLA task.
- dataloader.py
- train.py
- utils.py

---
## Dataset
[EndoVis17/18-VQLA-Extended](https://drive.google.com/file/d/1-FXOdhD3uw55ATDgI1wPEe-txyuCiP2E/view?usp=drive_link).

---

## Run training
- Train on EndoVis-18-VLQA-Extended
    ```bash
    python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver cat --batch_size 32 --epochs 80 --savelog /SAVELOG_PATH/ --detloss giou --claloss focal --uncer True
    ```
---
## Evaluation
- Evaluate on EndoVis17/18-VQLA-Extended
    ```bash
    python train.py --validate True --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver cat --batch_size 32
    ```
