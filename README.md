# JIT-Context: Context-Aware Just-In-Time Defect Prediction

This repository contains the implementation of our context-aware Just-In-Time Defect Prediction framework.

---

## 🖥️ Environment

- **Operating System:** Ubuntu 20.04  
- **GPU:** NVIDIA V100 (single GPU)  
- **Python Environment:** Conda  

---

## 🔧 Environment Setup

### 1. Create Conda Environment

Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate jitcontext
```

---

### 2. Install Tree-sitter (Java Parser)

Clone the Tree-sitter Java repository:

```bash
git clone https://github.com/tree-sitter/tree-sitter-java
```

---

### 3. Install Python Dependencies

Install required Python packages:

```bash
pip install -r requirement.txt
```

---

### 4. Unzip Dataset Files

Unzip all `.zip` files in the root directory:

```bash
unzip *.zip .
```

---

## 🚀 Training & Testing

### Within-Project Training and Testing

```bash
python JIT-Context/run.py
```

---

### Cross-Project Training and Testing

```bash
python JIT-Context/cross_project_run.py
```

---

## 📊 Reconstruct Graph-Structured Data

If you want to rebuild the graph-structured commit data from scratch:

1. Create a folder named `Dataset` in the root directory:

```bash
mkdir Dataset
```

2. Inside `Dataset`, clone the project repositories used in the paper.


```bash
cd DataSet/
git clone https://github.com/apache/commons-math.git
git clone https://github.com/apache/commons-lang.git
git clone https://github.com/apache/commons-configuration.git
git clone https://github.com/apache/commons-collections.git
git clone https://github.com/apache/ant-ivy.git
git clone https://github.com/apache/commons-compress.git
git clone https://github.com/apache/commons-io.git
git clone https://github.com/apache/commons-net.git
git clone https://github.com/apache/parquet-java parquet-mr
git clone https://github.com/apache/commons-vfs.git
git clone https://github.com/apache/opennlp.git
git clone https://github.com/apache/commons-digester.git
git clone https://github.com/apache/commons-dbcp.git
git clone https://github.com/apache/giraph.git
git clone https://github.com/apache/commons-jcs.git
git clone https://github.com/apache/commons-bcel.git
git clone https://github.com/apache/commons-codec.git
git clone https://github.com/apache/commons-beanutils.git
git clone https://github.com/apache/commons-validator.git
git clone https://github.com/apache/gora.git
git clone https://github.com/apache/commons-scxml.git
```
3. Run the preprocessing script:

```bash
python JIT-DP/util/process_commit.py
```

This will reconstruct the graph-based contextual data.

---

## 🤖 Running LLM-Based Baselines

After preparing the `Dataset` folder:

### Run LLM-Simple

```bash
python LLM_Simple.py
```

### Run LLM-Context

```bash
python LLM_Context.py
```

### Evaluate LLM Results

```bash
python evaluation.py
```

---
