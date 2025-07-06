# Setup
1. Clone this repository.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Download data from https://www.kaggle.com/datasets/hungbui0416/cifar-10

# Usage
- To train a supervised learning model:
  ```powershell
  python supervised_learning.py
  ```
- To evaluate a model:
  ```powershell
  python evaluate.py --model sl
  ```
- To train with contrastive learning:
  ```powershell
  python contrastive_learning.py
  ```
- To train the contrastive learning model for supervised learning:
  ```powershell
  python supervised_learning.py --model cl_scl 
  ```
- To train the contrastive learning model for hard negative contrastive learning:
  ```powershell
  python hardneg_contrastive_learning.py --sl_model sl_cl_scl --cl_model cl_scl
  ```
- To train the hard negative contrastive learning model for supervised learning:
  ```powershell
  python supervised_learning.py --model hcl_scl
  ```
- To classify an image:
  ```powershell
  python demo1.py
  ```
- To get misclassified images:
  ```powershell
  python demo2.py
  ```
