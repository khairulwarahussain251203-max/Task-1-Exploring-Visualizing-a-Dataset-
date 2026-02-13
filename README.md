# Exploring and Visualizing a Simple Dataset 
AI/ML-Internship_Task-1

# Iris Dataset Analysis - Exploratory Data Analysis & Visualization


## Task Objective

The primary objective of this project is to **load, inspect, analyze, and visualize** the classic Iris dataset to understand:
- Data structure and characteristics
- Statistical properties and distributions
- Relationships between different features
- Patterns that distinguish between different species
- Outlier detection and data quality assessment

This is a fundamental data science task that demonstrates the complete exploratory data analysis (EDA) workflow.

---

## Dataset Description

### **Dataset:** Iris Flower Dataset
**Source:** UCI Machine Learning Repository / Seaborn built-in dataset

### **Dataset Characteristics:**
| Attribute | Details |
|-----------|---------|
| **Samples** | 150 instances |
| **Features** | 4 numerical features + 1 categorical target |
| **Classes** | 3 species (50 samples each) |
| **Missing Values** | None |
| **Balanced** | Yes (perfectly balanced) |

### **Features:**
1. **Sepal Length** (cm) - Continuous
2. **Sepal Width** (cm) - Continuous  
3. **Petal Length** (cm) - Continuous
4. **Petal Width** (cm) - Continuous
5. **Species** - Categorical (Setosa, Versicolor, Virginica)

---

## 🛠 Tools & Libraries Used

| Library | Purpose |
|---------|---------|
| **Pandas** | Data loading, manipulation, and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Basic plotting and visualizations |
| **Seaborn** | Statistical data visualization |
| **Jupyter** | Interactive development environment |

---

## Methodology

### **Phase 1: Data Loading & Initial Inspection**
- Loaded dataset using Seaborn's built-in function
- Checked shape, columns, data types
- Examined first/last rows and random samples

### **Phase 2: Data Exploration & Statistics**
- Generated summary statistics (mean, std, min, max, quartiles)
- Grouped analysis by species
- Checked for missing values and duplicates
- Analyzed species distribution

### **Phase 3: Data Visualization**
1. **Scatter Plots** - Feature relationships
2. **Pairplot** - Multi-feature relationships
3. **Histograms** - Distribution analysis
4. **Box Plots** - Outlier detection
5. **Correlation Heatmap** - Feature correlations
6. **Violin Plots** - Distribution density
7. **Swarm Plots** - Detailed distribution
8. **Count Plots** - Categorical distribution

---

## Models Applied

**Note:** This is primarily an **Exploratory Data Analysis (EDA)** project. While no predictive models were built in this specific notebook, the analysis provides insights that would inform model selection:

### **Recommended Models for This Dataset:**

| Model | Purpose | Why Suitable |
|-------|---------|--------------|
| **Logistic Regression** | Classification | Linear separability of Setosa |
| **Decision Trees** | Classification | Interpretable, handles non-linearity |
| **Random Forest** | Classification | Robust, handles feature interactions |
| **SVM (RBF kernel)** | Classification | Excellent for small datasets |
| **K-Nearest Neighbors** | Classification | Simple, effective for Iris |
| **LDA/QDA** | Classification | Natural fit for Gaussian distributions |

### **Performance Metrics Used:**
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC

---

## Key Results & Findings

### **1. Dataset Overview**
✓ Total Samples: 150
✓ Features: 4 numerical + 1 categorical
✓ Species: Setosa (50), Versicolor (50), Virginica (50)
✓ Missing Values: 0
✓ Duplicates: 0
✓ Dataset Balance: Perfectly balanced (33.33% each)


### **2. Statistical Insights**

| Feature | Range (cm) | Mean ± Std | Key Observation |
|---------|-----------|------------|-----------------|
| **Sepal Length** | 4.3 - 7.9 | 5.84 ± 0.83 | Moderate variation |
| **Sepal Width** | 2.0 - 4.4 | 3.05 ± 0.43 | Some outliers present |
| **Petal Length** | 1.0 - 6.9 | 3.76 ± 1.76 | High discriminative power |
| **Petal Width** | 0.1 - 2.5 | 1.20 ± 0.76 | High discriminative power |

### **3. Species Characteristics**

| Species | Sepal Length | Sepal Width | Petal Length | Petal Width | Distinguishability |
|---------|--------------|-------------|--------------|-------------|-------------------|
| **Setosa** | Small-Medium | Large | Very Small | Very Small | **Easily separable** |
| **Versicolor** | Medium | Medium | Medium | Medium | Overlaps with Virginica |
| **Virginica** | Large | Medium-Large | Large | Large | Overlaps with Versicolor |

### **4. Correlation Analysis**

**Strong Correlations:**
- ✅ **Petal Length ↔ Petal Width**: **0.96** (Very strong positive)
- ✅ **Petal Length ↔ Sepal Length**: **0.87** (Strong positive)
- ✅ **Petal Width ↔ Sepal Length**: **0.82** (Strong positive)

**Weak Correlations:**
- ❌ **Sepal Length ↔ Sepal Width**: **-0.12** (Very weak negative)
- ❌ **Sepal Width ↔ Petal Length**: **-0.43** (Moderate negative)
- ❌ **Sepal Width ↔ Petal Width**: **-0.37** (Weak negative)

### **5. Outlier Detection**

| Feature | Outliers Present | Species Affected | Severity |
|---------|-----------------|------------------|----------|
| **Sepal Width** | Yes | Versicolor, Virginica | Minor (2-3 points) |
| **Sepal Length** | No | - | None |
| **Petal Length** | No | - | None |
| **Petal Width** | No | - | None |

---

## Visualizations

### **Generated Visualizations:**

1. **Scatter Plot Matrix** - Shows pairwise feature relationships colored by species
2. **Distribution Histograms** - Frequency distribution of each feature
3. **Box Plots** - Statistical summary and outlier visualization
4. **Correlation Heatmap** - Feature correlation strength and direction
5. **Violin Plots** - Probability density of feature distributions
6. **Swarm Plots** - Individual data point distribution
7. **Count Plot** - Species balance visualization

### **Sample Visual Output:**
![Iris Pairplot](images/iris_pairplot.png)
*Figure 1: Pairwise relationships between all features*

![Correlation Heatmap](images/iris_correlation.png)
*Figure 2: Feature correlation heatmap*

---

## Installation & Setup

### **Prerequisites:**
```bash
Python 3.8+
pip or conda package manager
VS Code (recommended) or Jupyter Notebook