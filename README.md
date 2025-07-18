# Atomistic Phase Transition Mechanism and Its Relationship with the High Thermoelectric Performance in SnSe

This repository supports the research article:

**"Atomistic Phase Transition Mechanism and Its Relationship with the High Thermoelectric Performance in SnSe"**

## Repository Structure

- **`UDVD/`**:  
  A deep learning model for **unsupervised denoising of STEM videos**, based on the method proposed by Sheth *et al.*.  
  The model is retrained on STEM videos from this project.

- **`U-Net/`**:  
  A U-Net-based network for **atom recognition** in HAADF-STEM images.  
  Suitable for a wide range of materials.  
  Trained model: `U-Net/output/model.pth`.

- **`Atomic-Segmentation/`**:  
  A specific application for **SnSe phase analysis**, based on **angle measurements between Sn atoms**.  
  > To use this module, copy the trained model file `U-Net/output/model.pth` to the `Atomic-Segmentation/` directory.

## Recommended Environment

It is recommended to use the following environment (via `conda`) to ensure compatibility:

- **Python**: 3.10  
- **PyTorch**: 2.2.2 + CUDA 11.8  
- **NumPy**: 1.26 (⚠️ PyTorch 2.2.2 is incompatible with NumPy ≥ 2.0)  
- **OpenCV**: 4.8.1  
- **Others**:  
  - SciPy 1.11  
  - scikit-learn 1.3  
  - pandas  
  - scikit-image  
  - matplotlib  
  - tensorboard  
  - jupyter  
  - openpyxl  

You may use `conda` to install and manage these dependencies.

---

## Contact

For questions or contributions, please contact:  
**Ziyang Huang** (huangziyang02@gmail.com)

---
