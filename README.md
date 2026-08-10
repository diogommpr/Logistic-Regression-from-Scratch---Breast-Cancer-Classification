# Logistic-Regression-from-Scratch---Breast-Cancer-Classification

## 1. Overview
This project implements binary logistic regression from scratch using NumPy to classify breast tumors as benign or malignant using the Breast Cancer Wisconsin dataset.
The objective was to develop a practical understanding of the mathematical foundations of logistic regression by implementing the core algorithm directly, instead of relying on a pre-built machine learning estimator such as scikit-learn's LogisticRegression.
The project covers the main components of logistic regression, including the sigmoid function, gradient computation, gradient descent, classification, and model evaluation. 
This project was developed based on concepts learned in Andrew Ng's _Supervised Machine Learning: Regression and Classification Course_. 

## 2. Dataset
The Breast Cancer Wisconsin dataset was used. The dataset contains 569 observations describing characteristics of cell nuclei extracted from digitized images of breast mass fine-needle aspirates.

The target variable is:

- `diagnosis`
  - `0` = Benign
  - `1` = Malignant

Although the dataset contains multiple numerical features, only two were selected for this implementation:

- `radius_mean`
- `texture_mean`

Using two features was intended to allow the observations and the learned decision boundary to be visualized in two dimensions while keeping the focus on understanding the logistic regression algorithm.

## 3. Machine Learning Workflow

### 3.1 Target Encoding

The original diagnostic labels were converted to binary values:

- Benign (`B`) → 0
- Malignant (`M`) → 1


### 3.2 Feature Selection

Mean radius and mean texture were selected as the two predictor variables.


### 3.3 Train-Test Split

The dataset was divided into training and testing subsets:

- 80% training data
- 20% test data
- `random_state=42` for reproducibility
- Stratification by diagnosis to preserve class proportions

### 3.4 Feature Scaling

Features were standardized using:

x_scaled = (x - mean) / standard deviation

The mean and standard deviation were calculated exclusively from the training data and subsequently applied to both the training and test sets to prevent data leakage.

## 4. Logistic Regression Implementation

The logistic regression model was implemented from scratch using NumPy.

### 4.1 Sigmoid Function

The sigmoid function converts the linear model output into a value between 0 and 1, which can be interpreted as the estimated probability of the positive class.

### 4.2 Cost Function

The logistic regression cost function was implemented to measure the error between the model predictions and the actual labels.

### 4.3 Gradient Computation

The gradients of the cost function were calculated for the model parameters `w` and `b`. These gradients were then used by gradient descent to update the parameters and reduce the cost.

### 4.4 Gradient Descent

Gradient descent was implemented to iteratively update the model parameters and minimize the cost function:

w = w - alpha * dJ/dw

b = b - alpha * dJ/db

An initial learning rate of 0.001 resulted in slow convergence. Increasing the learning rate to 0.1 allowed the cost function to converge much faster.

### 4.5 Classification

Predicted probabilities were converted into binary classifications using a threshold of 0.5:

- Probability ≥ 0.5 → Malignant
- Probability < 0.5 → Benign

## 5. Model Evaluation

The final model achieved:

| Dataset | Accuracy |
|---|---:|
| Training set | 89.45% |
| Test set | 86.84% |

The relatively small difference between training and test accuracy suggests that the model generalizes reasonably well to unseen observations within this dataset.

## 6. Visualization

The project includes visualizations of:

- The distribution of benign and malignant observations according to the two selected features
- The learning curve during gradient descent
- The effect of increasing the learning rate on convergence
- The learned logistic regression decision boundary

## 7. Key Findings

- Logistic regression can be implemented using a relatively small number of mathematical components: a linear function, sigmoid transformation, logistic loss, gradient computation, and iterative parameter updates.
- Increasing the learning rate from 0.001 to 0.1 substantially accelerated gradient descent convergence.
- Using only mean radius and mean texture, the model achieved 89.45% training accuracy and 86.84% test accuracy.
- Restricting the model to two features facilitates interpretation and visualization but likely limits its predictive capacity compared with models using the complete set of available features.

## 8. Tools and Technologies

- Python
- NumPy
- pandas
- matplotlib
- scikit-learn (`train_test_split` only)
- Jupyter Notebook

## 9. Repository Structure

    data/
        Breast Cancer Wisconsin.csv
    notebooks/
        01_logistic_regression.ipynb
    README.md

## 10. How to Reproduce

1. Clone the repository.
2. Place the dataset in the `data/` directory.
3. Open `01_logistic_regression.ipynb`.
4. Run all cells in sequence.



