# A Real-Time Cardiac Decision Support System for IoMT Using Edge AI and DDS

This repository provides the implementation for a decentralized medical decision-support system (MDSS). The architecture integrates **MobileNetV3** for demographic-specific edge inference and **RTI Connext DDS** for deterministic, real-time data distribution.

## System Overview
The system follows a three-stage communication pipeline:
1. **ECG Sensor Node (Publisher):** Streams preprocessed ECG data from the Adult or Pediatric datasets.
2. **ECG Inference Node (Subscriber & Publisher):** - Acts as a **Subscriber** to receive raw ECG data.
   - Performs real-time classification using the trained MobileNetV3 models.
   - Acts as a **Publisher** to send the resulting "Inference Alert" to the network.
3. **Nurse Alert Display (Final Subscriber):** Receives the processed alerts for real-time monitoring and clinical display.

> **Note:** If the Inference Node is not running or the Publisher is inactive, no data will be displayed on the Nurse Alert station, ensuring the system reflects the real-time state of the network.

---

## Repository Structure
* `Datasets/`: Preprocessed ECG images (Adults and Children).
* `Training_Models/`: Python scripts for training MobileNetV3 with history tracking.
* `DDS with RTI/`: Middleware configuration (XML QoS profiles) and Python scripts.
* `Paper_Results/`: Organized figures (Adult, Children, Ethernet, Wireless, Energy) representing the experimental findings.

---

## Setup and Reproducibility

### 1. Prerequisites
* **Python 3.x** (with `torch`, `torchvision`, `tqdm`).
* **RTI Connext DDS Pro 7.6.0** (or compatible).
* **Hardware:** PC with NVIDIA GPU (Inference Node) and Raspberry Pi 5 (Nurse Station).

### 2. Path Configuration (Important)
Before running the simulations, you **must** update the `data_dir` or file paths in the Python scripts. The current scripts are configured for a specific local directory. Ensure the path points to wherever you have saved the `Datasets/` folder on your machine:
```python
# Example of what to check in the .py files:
data_dir = r'C:\Path\To\Your\Local\Datasets'
