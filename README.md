# Multiple Instance Learning (MIL) Branch for Medical Image Processing

## Introduction

The MIL branch of this project addresses critical questions in dermoscopy image analysis by focusing on the relevance and spatial positioning of patches within an image. It is a part of a larger pipeline that includes an EViT branch and operates on medical images. This branch is crucial for understanding the role of patches in cancer diagnosis through weakly supervised learning.

The MIL branch comprises two fundamental components: a deep patch extractor and a MIL classifier. In the following sections, we will explore the constituent blocks of the MIL branch in more detail. Our approach introduces a two-step method for generalizing binary MIL classifiers and extends it to a three-step method for multi-class classification.

## MIL Classifier Architecture

### Feature Extractor

The first component of the MIL branch is a deep patch extractor denoted as E, responsible for generating
a 14 × 14 feature map.

![Feature Extractor Pipeline](MIL_Feature_extractor_v3.png)


Several options for feature extractors are integrated into the model, allowing for flexibility and adaptability to different types of image data.

### Binary Classification

For binary classification, the model employs a two-step approach that operates on a bag of embedded instances (X ∈ R^(N×D)). It utilizes two key functions:
- A non-linear classifier `h`, representing an instance-level classifier, which provides the probability of the positive class for each patch.
- A permutation-invariant aggregation function `ϕ`, which combines individual patch probabilities into a bag probability.

The order of these two function determines the type of MIL classifier: instance-level or embedding-level.

![Feature Extractor Pipeline](MIL_approaches_v3.png)

### Multi-Class Classification

The three-step method for multi-class MIL extends the binary approach with the following functions:
- `z`: A linear layer mapping the input to the number of classes.
- `ϕ`: A permutation-invariant aggregation function.
- `σ`: The softmax function, which, in binary scenarios, is replaced by a sigmoid activation function.

Both approaches consider the permutation invariance required for MIL models, accommodating various pooling functions like max, avg, and topk.

## Visualization of Pooling Functions

The model allows visualization of the different pooling functions: max, avg, and topk. These visualizations can provide insights into the areas within the patches that are most indicative of skin cancer.

![Feature Extractor Pipeline](MIL-Pools_Vis.png)

## Image Processing Capability

The MIL model is designed to process dermoscopic RGB images and greyscale mammography images, supporting publicly available datasets such as ISIC2019, PH2, Derm7pt, DDSM, an others. 


## Usage

To use this MIL model branch in your pipeline, follow these steps:

1. **Clone the Repository**
    - Clone this repository to your local machine to get started.
    ```bash
    git clone <repository-url>
    ```

2. **Set Up the Environment**
    - Follow the installation instructions to set up the necessary environment and dependencies.
        
        ```bash
        pip install -r requirements.txt
        ```

3. **Operational Modes**
    - Medical EViT supports various modes of operation, catering to different stages of model usage and analysis:

        **Training**
        - To train the model from scratch on your dataset.
        ```bash
        python main.py --mode train --dataset <your-dataset>
        ```

        **Fine-Tuning**
        - To fine-tune the pre-trained model on a specific dataset.
        ```bash
        python main.py --mode finetune --pretrained <path-to-pretrained-model> --dataset <your-dataset>
        ```

        **Testing**
        - To evaluate the model's performance on a test dataset.
        ```bash
        python main.py --mode test --checkpoint <path-to-model-checkpoint> --dataset <test-dataset>
        ```

        **Visualizing Heat Maps**
        - To visualize the patches that the MIL model identified as the most relevant, highlighting how the model focuses on different parts of the image.
        ```bash
        python main.py --mode visualize --checkpoint <path-to-model-checkpoint> --dataset <visualization-dataset>
        ```

    Replace `<repository-url>`, `<your-dataset>`, `<path-to-pretrained-model>`, `<path-to-model-checkpoint>`, and `<visualization-dataset>` with the actual values relevant to your project.

For a detailed understanding of the MIL branch and its role in the larger context, please refer to the extended abstract and paper associated with this repository.