# 🧑‍💻 Customer Churn Prediction  

Customer churn prediction using **Logistic Regression, Random Forest, and XGBoost with SMOTE balancing**.  
This project identifies customers likely to churn, helping businesses take proactive steps to improve retention.  

---

## 📂 Project Overview  

- Performed **data preprocessing** (handling missing values, encoding categorical variables).  
- Applied **SMOTE** to balance imbalanced classes.  
- Trained **multiple ML models** (Logistic Regression, Random Forest, XGBoost).  
- Evaluated models using **accuracy, precision, recall, F1-score, and ROC-AUC**.  
- Saved the best model (`rf_churn_model.joblib`) for deployment.  

---

## 🚀 Business Impact  

- **Customer Retention:** Identify customers at risk of leaving and take action to retain them.  
- **Revenue Growth:** Reducing churn leads to more stable and predictable revenue.  
- **Better Marketing:** Enables targeted offers for at-risk customers instead of broad campaigns.  
- **Operational Efficiency:** Focuses resources on customers who matter most.  

---

## 📊 Feature Importance Example  

The top features influencing churn:  

- Contract type  
- Monthly charges  
- Tenure  
- Payment method  

*(Visuals are generated in the script).*  

---

## ⚡ How to Run  

1. Clone the repository
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Install dependencies
pip install -r requirements.txt

3. Run the churn analysis script
python churn_analysis.py

4. The best-performing model will be saved as:
rf_churn_model.joblib


| Model               | Accuracy | ROC-AUC | Precision | Recall | F1-score |
| ------------------- | -------- | ------- | --------- | ------ | -------- |
| Logistic Regression | \~XX%    | \~XX%   | \~XX%     | \~XX%  | \~XX%    |
| Random Forest       | \~XX%    | \~XX%   | \~XX%     | \~XX%  | \~XX%    |
| XGBoost             | \~XX%    | \~XX%   | \~XX%     | \~XX%  | \~XX%    |

➡️ In our experiments, XGBoost/Random Forest performed best.

## 📊 Feature Importance Example
The top features influencing churn:
- **Contract type**  
- **Monthly charges**  
- **Tenure**  
- **Payment method**  

*(Visuals are generated in the script.)*  

---

## 🌟 Future Improvements
- Deploy as a **Flask/Streamlit app**  
- Perform **hyperparameter tuning** with GridSearchCV  
- Deploy the **best model as an API** for production  

## 💼 Business Impact
- Helps companies **identify customers at risk of leaving**, enabling proactive retention strategies.  
- Improves **customer satisfaction** by addressing issues before they cause churn.  
- Reduces **revenue loss** by focusing marketing and support efforts on high-risk customers.  
- Provides **data-driven decision making** for sales and customer service teams.  
- Can be adapted across industries like **telecom, banking, e-commerce, and SaaS businesses**.  

## 📜 License  
This project is licensed under the **MIT License** – you are free to use, modify, and distribute it with proper attribution.  




