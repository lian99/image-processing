# 🖼️ Image Processing — Fourier & Enhancement (HW4)

## 📌 Overview

This project explores key concepts in image processing, focusing on **Fourier transforms**, **frequency-domain scaling**, and **image restoration techniques**.

The implementation demonstrates how frequency-domain operations affect images and applies various enhancement methods to improve degraded images.

---

## 🧠 Topics Covered

* Fourier Transform (FFT)
* Frequency-domain image scaling
* Image enhancement and restoration
* Filtering techniques (spatial & frequency domain)

---

## 📂 Project Structure

```
q2/
 ├── zebra.py
 ├── zebra.jpg
 └── zebra_scaled.png

q3/
 ├── main.py
 ├── BadImagesFixing.py
 └── images/
```

---

## 🧪 Part 1 — Fourier Transform (Theory)

This section explores theoretical aspects of Fourier transforms, including:

* Comparison between spatial scaling and frequency-domain scaling
* Mathematical properties of Fourier transform uniqueness

👉 Full explanations are included in the report.

---

## 🦓 Part 2 — Image Scaling in Frequency Domain

Implemented two different scaling approaches using Fourier transforms:

### 1. Zero Padding

* Expands the Fourier domain
* Results in smoother interpolation
* Produces a larger image (2H × 2W)

### 2. Fourier Scaling Formula

* Applies transformation:

  F[f(ax, by)] = (1 / |ab|) F(u/a, v/b)

* Produces repeated patterns (tiling effect)

---

### 🔍 Key Observations

* Zero padding preserves image structure better
* Fourier scaling introduces multiple copies of the image
* Frequency-domain manipulation directly impacts spatial results

---

## 🛠️ Part 3 — Image Restoration

Enhanced and restored multiple degraded images using different techniques.

According to the assignment , images include noise, blur, low contrast, and frequency artifacts.

---

## 🔧 Techniques Used

* Histogram Equalization
* Gamma Correction
* Median / Bilateral Filtering
* Sharpening filters
* Fourier domain filtering

---

## 📸 Examples of Improvements

### 🧸 Baby Image

* Combined multiple images to reduce noise
* Improved clarity using filtering

### 🌬️ Windmill

* Removed periodic noise using frequency filtering

### 🍉 Watermelon

* Applied sharpening filters to enhance edges

### ☂️ Umbrella

* Corrected ghosting effect caused by image averaging

### 🇺🇸 USA Flag

* Removed noise and restored details

### 🏠 House

* Reduced motion blur from multiple shifted images

### 🐻 Bears

* Enhanced brightness and contrast (low gray values)

---

## 💡 Key Learnings

* Frequency-domain techniques are powerful for noise removal
* Different types of degradation require different filters
* Combining spatial and frequency methods gives better results
* Fourier transform provides insight into image structure

---

## 🚀 How to Run

### Part 2 — Zebra Scaling

```bash
python zebra.py
```

---

### Part 3 — Image Fixing

```bash
python main.py
```

---

## 📊 Results

* Successfully scaled images using Fourier techniques
* Restored multiple degraded images using appropriate filters
* Demonstrated practical understanding of frequency-domain processing

---

