# 🧠 Hybrid Images

Hybrid Images is a technique that combines low-frequency information from one image and high-frequency information from another to create a "hybrid" image that appears differently depending on the viewing distance.

## 📚 Lý thuyết

Hybrid Images technique introduced in *"Hybrid Images" of Oliva, Torralba, and Schyns (2006)*.

---

## 🧩 Tuition

- An image **containing low frequency components** of a raw (`Image1`)
- An image **containing high frequency components** of a raw (`Image2`)
- Combining two images into an image (`Hybrid`), for:
  - Near sight of view → see the high-freq image clearly (`Image2`)
  - Far sight of view → see the low-freq image clearly (`Image1`)

---

## 🧮 Steps and Calculations

### 1. Gaussian Filter (Low-pass)

Áp dụng bộ lọc Gaussian để làm mờ ảnh:

```math
G_{\sigma}(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
```

### 2. Low Pass filter
Cross correlation or Convolution 2D each channels of the image with Gaussian filter
```math
Low-passed = I * G
```

### 3. High Pass filter
Take the the image minus the the low-pass component
```math 
High-passed = I - Low-passed
``` 

### 4. Hybrid Images
Sum low-pass and high-pass components together 
```math
Hybrid = Low-passed + High-passed
```

