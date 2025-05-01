
---

# 👁️ Eye Disease Classification with CNN

This project focuses on detecting common eye conditions — **Cataract**, **Glaucoma**, and **Normal** — using a Convolutional Neural Network built with TensorFlow/Keras.

The idea is to feed the model with eye images and have it predict the type of condition present, if any. This could be a useful tool for initial screenings or as a learning reference for deep learning applications in healthcare.

---

## 📂 Dataset Structure

The dataset is organized in this format:

```
Data/
├── train/
│   ├── cataract/
│   ├── glaucoma/
│   └── normal/
├── valid/
│   ├── cataract/
│   ├── glaucoma/
│   └── normal/
```

- **Training set**: ~3,100 images  
- **Validation set**: ~1,200 images  
- Images are resized to **128x128** before training.

---

## 🧠 Model Overview

The model uses several convolutional layers, each followed by activation and pooling:

- Multiple `Conv2D` layers with increasing filters: from 32 to 512
- MaxPooling after every couple of convolution layers
- Dropout layers to reduce overfitting
- Dense layer with 512 units before the final softmax output

It’s trained using **categorical crossentropy loss** and the **Adam** optimizer.

---

## 📊 Results

- **Final training accuracy**: ~85%  
- **Validation accuracy**: ~77%  
- The model performs well across all three classes, though precision varies slightly.

Here's a quick breakdown:

| Class     | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| Cataract  | 0.93      | 0.74   | 0.82     |
| Glaucoma  | 0.73      | 0.74   | 0.73     |
| Normal    | 0.70      | 0.83   | 0.76     |

A confusion matrix and accuracy plots are also included in the notebook/scripts.

---

## 🖼️ Making Predictions

You can use the trained model to classify a new image:

1. Load the image using OpenCV or Keras utilities
2. Preprocess it (resize, normalize, batch)
3. Pass it to the model to get predictions
4. Display the result with `matplotlib`

Example prediction:
```python
image = load_img('path/to/image.jpg', target_size=(128,128))
input_arr = img_to_array(image)
input_arr = np.expand_dims(input_arr, axis=0)
pred = model.predict(input_arr)
```

---

## 💾 Model Export

Once training is done, the model is saved as:

```
trained_eyes_disease_model.keras
```

You can load it later with:
```python
model = tf.keras.models.load_model('trained_eyes_disease_model.keras')
```

---

## 🔧 Requirements

- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn

---
