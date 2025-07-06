# CIFAR-10 Classification

This project implements various deep learning approaches for image classification on the CIFAR-10 dataset, including supervised learning, contrastive learning, and hard negative contrastive learning.

## Project Structure
- `contrastive_learning.py`, `hardneg_contrastive_learning.py`: Implementations of contrastive learning methods.
- `supervised_learning.py`: Supervised learning training routines.
- `model.py`: Model architectures.
- `data_utils.py`: Data loading and augmentation utilities.
- `contrastive_loss.py`: Loss functions for contrastive learning.
- `train_test_utils.py`: Training and testing utilities.
- `evaluate.py`: Evaluation scripts.
- `demo1.py`, `demo2.py`: Example/demo scripts.
- `requirements.txt`: Python dependencies.
- `data/`: Contains CIFAR-10 dataset (train/test split by class).
- `models/`, `final_models/`: Saved model checkpoints.
- `notebooks/`: Jupyter notebooks for experiments and analysis.
- `misclassified/`: Misclassified images for analysis.

## Setup
1. Clone this repository.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Ensure the CIFAR-10 dataset is available in the `data/` directory (already included in this repo structure).

## Usage
- To train a supervised model:
  ```powershell
  python supervised_learning.py
  ```
- To train with contrastive learning:
  ```powershell
  python contrastive_learning.py
  ```
- To evaluate a model:
  ```powershell
  python evaluate.py
  ```
- For more details, see the Jupyter notebooks in `notebooks/`.

## Results
- Trained models are saved in `models/` and `final_models/`.
- Misclassified images are saved in `misclassified/` for further analysis.

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- SimCLR, Supervised Contrastive Learning, and related papers.

## License
This project is for educational and research purposes.
