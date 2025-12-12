# Food-101 Image Classification
### *Advanced Machine Learning Project — Boston University*

---

## 1. Project Overview

This project applies deep learning and transfer learning methods to the **Food-101 dataset**, a large-scale image classification benchmark of 101 food categories.  
The goal was to evaluate different convolutional neural network (CNN) architectures, regularization strategies, and transfer learning methods to determine which approach performs best under realistic compute constraints.

We built and compared:

- A baseline CNN  
- An improved custom CNN (Model 2)  
- Transfer learning with EfficientNetB0  
- Transfer learning with MobileNetV2  
- Multiple augmentation and regularization strategies  

The final deliverable identifies the most effective model for classification and explains why other approaches failed to converge.

---

## 2. Dataset Description

We used the **Food-101 dataset**, but instead of using the official predefined split, we implemented our own **80/10/10 split**:

- **80% training**
- **10% validation**
- **10% test**

This ensured:

- Consistent evaluation across all models  
- Fair comparison of architectures  
- Proper early stopping and hyperparameter tuning  

### Dataset Properties

- 101 food categories  
- ~1000 images per category (~100K images total)  
- High variability in lighting, angle, plating, and backgrounds  
- Real-world noise (occlusions, blur, clutter)

### Practical Challenges

- Large dataset increases memory usage  
- Many classes → higher chance of misclassification  
- Shallow models underfit severely  
- Transfer learning requires correct preprocessing & compute power  

---

## 3. Baseline Model — Custom CNN (Model 1)

### Architecture
A simple CNN with 2–3 convolutional layers and max pooling.

### Result
- **~1% accuracy** across train/validation  
- Model did not learn meaningful feature representations  
- Demonstrated severe underfitting

This confirmed that Food-101 requires deeper models.

---

## 4. Improved Custom CNN (Model 2)

Model 2 introduced:

- More convolutional layers  
- Increased filters  
- Batch normalization  
- Larger dense layers  
- Dropout & regularization  
- Strong augmentation pipeline  

### Performance  
- **43.78% validation accuracy**  
- **Only 413K trainable parameters**  
- Trained for **27 epochs (~2–3 hours)**  
- Clear learning curve improvement over Model 1

This became the **best model overall**.

---

## 5. Transfer Learning — EfficientNetB0

Three strategies were tested:

1. **Frozen base model**  
2. **Fully unfrozen base**  
3. **Partially unfrozen (fine-tuning last blocks)**  

### Results  
All versions produced **~1% accuracy**, showing:

- Training was not converging  
- Features were not adapting  
- Preprocessing mismatch or insufficient compute was likely  
- EfficientNetB0 was not feasible under course hardware constraints  

---

## 6. Transfer Learning — MobileNetV2

MobileNetV2 was tested under the same three configurations.

### Results  
All configurations also produced **~1% accuracy**, indicating:

- Features were not learned  
- Training setup incompatible with model expectations  
- Strong dependence on correct input scaling and batch normalization  
- Resource constraints stopped convergence  

---

## 7. Regularization & Data Augmentation

To limit overfitting and improve generalization:

- Random rotation  
- Horizontal & vertical flips  
- Random zoom  
- Brightness and contrast adjustments  
- L2 regularization  
- Dropout  
- Early stopping  
- Learning rate scheduling  

These helped significantly in Model 2, but did not fix transfer learning failures.

---

## 8. Final Model Selection

### 🏆 **Final Model: Custom CNN (Model 2)**

| Model | Accuracy |
|-------|----------|
| Baseline CNN | ~1% |
| **Custom CNN Model 2** | **43.78%** |
| EfficientNetB0 | ~1% |
| MobileNetV2 | ~1% |

### Why Model 2 Won

- Deep enough to learn meaningful features  
- Lightweight enough to train under compute limits  
- Robust augmentation improved generalization  
- Maintained stable training behavior  
- Outperformed all transfer-learning attempts by a large margin  

---

## 9. Key Lessons Learned

- Shallow CNNs cannot learn complex, high-variance food images  
- Transfer learning is **not guaranteed** — mismatched preprocessing or insufficient compute prevents convergence  
- Effective data augmentation is essential for Food-101  
- Custom architectures can outperform “state-of-the-art” models under real resource constraints  
- 80/10/10 splitting ensures consistent, fair evaluation across all models  

---

## 10. Future Work

To push performance higher:

- Use TPU/GPU acceleration for EfficientNet or MobileNet  
- Experiment with **EfficientNetB3–B7** or **ResNet50**  
- Try **Vision Transformers (ViT)**  
- Use advanced augmentation (MixUp, CutMix)  
- Implement cosine learning rate warmup  
- Use class-balanced sampling or weighting  

---

## 📁 11. Files in This Folder

### 📓 Jupyter Notebooks  
(Links open fully rendered on GitHub)

- **Final Model Notebook**  
  [FinalCodeNotebook.ipynb](https://github.com/amarem07/Portfolio/blob/main/food-101-image-classification/FinalCodeNotebook.ipynb)

- **Milestone 1 Notebook**  
  [Milestone01.ipynb](https://github.com/amarem07/Portfolio/blob/main/food-101-image-classification/Milestone01.ipynb)

- **Milestone 2 Notebook**  
  [Milestone02.ipynb](https://github.com/amarem07/Portfolio/blob/main/food-101-image-classification/Milestone02.ipynb)

---

