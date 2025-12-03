# Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project applies **Unsupervised Machine Learning** techniques to segment customers based on their purchasing behavior. By analyzing **Annual Income** and **Spending Scores**, we identify distinct customer groups to enable targeted marketing strategies.

## 🛠️ Tech Stack
* **Python** (NumPy, Pandas)
* **Machine Learning** (Scikit-learn: K-Means, StandardScaler)
* **Visualization** (Matplotlib, Seaborn)

## 📊 Methodology

### 1. Data Preprocessing
Since K-Means is a distance-based algorithm, the data was normalized using **StandardScaler** to ensure equal weighting for Income and Spending features.

### 2. Finding the Optimal Clusters (Elbow Method)
The **Elbow Method** was used to determine the optimal number of clusters ($k$). The WCSS (Within-Cluster Sum of Squares) graph showed a clear "elbow" at **k=5**.

<img width="1000" height="500" alt="resim" src="https://github.com/user-attachments/assets/194f23db-6a95-4f63-ab62-bed4ec0d13ed" />


### 3. Model Results & Business Insights
The K-Means algorithm successfully identified **5 distinct customer segments**:

| Segment Profile | Income | Spending | Strategy Suggestion |
| :--- | :--- | :--- | :--- |
| **VIP / Champions** | High | High | Target with exclusive offers & loyalty programs. |
| **Potential Savers** | High | Low | Promote "Premium/Quality" products to encourage spending. |
| **Standard** | Medium | Medium | Maintain engagement with regular campaigns. |
| **Spenders** | Low | High | Offer payment plans or discounts (Budget-conscious but willing to spend). |
| **Economical** | Low | Low | Target only during major clearance sales. |

## 📈 Visualizations
Below is the final clustering of mall customers:

<img width="1000" height="600" alt="resim" src="https://github.com/user-attachments/assets/ca017473-bef9-4108-bb3b-ca377d330608" />


## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ulasaksac/Mall-Customer-Segmentation.git](https://github.com/ulasaksac/Mall-Customer-Segmentation.git)
