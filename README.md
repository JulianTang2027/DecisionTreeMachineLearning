# Romance Movie Ratings Analysis

## **Overview**
This project explores how various factors influence how individuals of different genders rate romance movies. Our goal is to quantify and predict the subjective preferences of viewers for romance movies by leveraging machine learning models. 

The project combines data from multiple sources and uses decision trees and neural networks to analyze and predict trends in movie ratings. By evaluating different factors—like age gaps between leads, gender representation, and genre-specific attributes—we aim to uncover key insights into audience preferences.

---

## **Data Sources**
We aggregated data from the following sources:
- **GroupLens**: Provides large datasets on movie ratings and reviews.
- **IMDb**: Supplies detailed movie attributes, including lead genders, directors, and age ratings.
- **Letterboxd**: Helps identify popular movies and observe trends based on community statistics.

---

## **Attributes Considered**
Here are the attributes we examined:
- **Age gap**: Difference in ages of the main two leads.
- **Age rating**: Movie ratings (e.g., G, PG-13, R).
- **Lead genders**: Gender representation of the first and second leads.
- **Writer/Director gender**: To analyze representation behind the camera.
- **IMDb star meter**: A proxy for movie popularity.
- **Bechdel test**: Whether the movie passes this gender-equality benchmark.
- **TV Tropes-related tags**: Captures narrative themes and clichés.
- **Genre**: Romantic sub-genres such as Romantic Comedy or Romantic Drama.

---

## **Methodology**
1. **Data Preprocessing**:
   - Removed incomplete or missing entries.
   - Normalized continuous variables (e.g., age gaps, IMDb ratings) using Z-score standardization for effective learning.
   - Mapped external data sources to unify attributes.

2. **Machine Learning Models**:
   - **Decision Tree (ID3)**: Built using a flexible algorithm to handle continuous and discrete variables. 
   - **Neural Network**: Used to capture non-linear relationships and complex patterns in the data.

3. **Evaluation Metrics**:
   - **Prediction accuracy** across gender categories.
   - **Precision-recall curves** for gender-specific analysis.
   - **F1 score** to balance precision and recall.
   - Statistical significance testing between the performance of decision trees and neural networks.

---

## **Key Files**
- **`MSE2.py`**: Contains the neural network implementation, using PyTorch, Pandas, NumPy, and Matplotlib for training and visualization.
- **`newID3.py`**: Implements the ID3 decision tree algorithm.
- **`trainfinalproject.py`**: Trains the decision tree model and uses the `pickle` library to save it for reuse.
- **`AccuracyTreePruneTree.py`**: Implements the pruning process for the decision tree to balance specificity and generalization.

---

## **Challenges and Insights**
One notable challenge was balancing the tradeoff between specificity and generalization during the pruning of the decision tree. The initial tree captured highly specific user preferences but lost accuracy when pruned due to the elimination of nuanced patterns critical for prediction.

---

## **Technologies Used**
- **Programming Languages**: Python
- **Libraries**:
  - PyTorch
  - Pandas
  - NumPy
  - Matplotlib
  - Pickle

---

## **How to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/JulianTang2027/DecisionTreeMachineLearning.git
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training scripts:
   - For the neural network:
     ```bash
     python MSE2.py
     ```
   - For the decision tree:
     ```bash
     python trainfinalproject.py
     ```
