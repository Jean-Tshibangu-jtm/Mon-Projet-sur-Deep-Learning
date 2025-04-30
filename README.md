
# Deep Learning Applied to Cybersecurity for Intrusion Detection

## 📌 Context

With the increasing complexity of IT networks and the proliferation of cyber threats, traditional intrusion detection tools (like Snort or Suricata) are reaching their limits. Their dependency on signatures makes it difficult to detect unknown (zero-day) attacks. In this context, **Deep Learning techniques** emerge as a powerful alternative, capable of identifying complex anomalies and learning from large-scale datasets.

## ❗ Problem Statement

How can we design an intrusion detection model that:
- Detects network attacks (e.g., **DDoS**) in **near real-time**, even if they are not listed in a signature database?
- Efficiently handles **massive volumes of heterogeneous network data**?
- **Improves detection accuracy** compared to traditional methods while **reducing false positives**?

## 🧪 Testbed Environment

The testbed is based on a simulated infrastructure consisting of two networks:
- An **Attack Network** generating various types of attacks (Brute Force SSH, DDoS, Botnet, etc.).
- A **Victim Network** capturing network traffic continuously for **five days**.

Network packets are processed using **CICFlowMeter**, which extracts over 80 statistical features.  
`.pcap` files are converted to `.csv` format via this Java tool, enabling bidirectional (BiFlow) flow analysis.

## 📂 Dataset Description

- **Source**: [CICIDS 2017 - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)  
- **Size**: 1,580,215 records  
- **Attack types**: DDoS, Brute Force, Botnet, Web Attack, Infiltration, etc.  
- **Focus of this study**: DDoS attacks detected on **Friday afternoon** (~225,745 samples)  
- **Format**: 83 columns with TCP/UDP flow features (duration, packet sizes, intervals, flags...)

## ⚙️ Technologies Used

- **Language**: Python  
- **Libraries**: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `Keras`, `TensorFlow`  
- **IDE**: Jupyter Notebook, Spyder  
- **Environment**: Anaconda, Google Colab  
- **Preprocessing**: CICFlowMeter, RFE for feature selection

## 🧠 AI & Algorithms Used

### 🔍 Preprocessing
- **Data cleaning and normalization**
- **Missing data analysis** (none found)
- **Feature selection** using **Recursive Feature Elimination (RFE)**, selecting:
  - `Flow-Duration`
  - `Flow IAT Std`
  - `Average Packet Size`
  - `Bwd Packet Length Std`

### 🏗️ Deep Learning Model
- **Architecture**: Keras Sequential model with several dense layers
- **Loss function**: `binary_crossentropy`
- **Optimizer**: `Adam`
- **Evaluation**: Confusion matrix, precision, recall, F1-score
- **Hyperparameters**:
  - `Epochs = 120`
  - `Batch size = 1`
  - `Metrics = accuracy`

## 📈 Results

| Metric              | Result      |
|---------------------|-------------|
| Accuracy            | **97%**     |
| DDoS Detection      | Very reliable |
| False Positives     | Low         |
| Main Advantage      | Detects complex anomalies without signatures |

## ✅ Contributions

- Built a **Deep Learning-based intrusion detection model**
- Used a **real and recent dataset** with diverse intrusion scenarios
- Compared performance with traditional tools (Snort, Suricata)
- Ensured reproducibility with complete Python scripts

## 🚀 Future Work

- Integrate a real-time system based on **Edge AI**
- Enhance performance using **CNN + LSTM**
- Extend detection to malware, spyware, and targeted infiltration
- Deploy into a **SIEM architecture** (Splunk, ELK)

---

## 📝 Important Note

> **N.B**: If your system is inadequate, I humbly ask you to stop here, as the program won't work efficiently and a lot of time will be wasted.

This project relies on **two main files**:
- `data_processing.py`: used for **data preprocessing**, **graphical analysis**, and **relevant feature selection**.
- `construction_DeepLearning.py`: contains the code for **building the deep learning model**.

➡️ Once the preprocessing is complete, you can **start training** the model.
