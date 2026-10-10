\# Smart House Price Prediction System



\## Project Overview



The Smart House Price Prediction System is a machine learning project that predicts house prices based on property-related features such as location, area, bedrooms, bathrooms, floors, parking, property age, and other attributes.



The project includes a trained machine learning model and an interactive Streamlit web application for house price prediction.



\## Features



\- House price prediction using Machine Learning

\- Linear Regression model

\- Data preprocessing and categorical encoding

\- Interactive Streamlit web application

\- Indian currency formatting (₹ Lakhs/Crore)

\- Property summary after prediction

\- Model performance metrics

\- Saved trained model and preprocessor



\## Dataset



\- Records: 2,500

\- Columns: 16

\- Target variable: `Price\_INR`



\## Machine Learning Model



The project uses \*\*Linear Regression\*\* for house price prediction.



\### Data Split



\- Training data: 80%

\- Testing data: 20%



\### Model Performance



\- R² Score: 96.58%

\- MAPE: 4.45%

\- MAE: ₹474,560.29

\- RMSE: ₹627,504.71



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- Matplotlib

\- Seaborn

\- Streamlit

\- Jupyter Notebook

\- VS Code

\- Anaconda



\## Project Structure



```text

ibmbob/

├── app/

│   └── app.py

├── datasets/

│   ├── house\_prices\_clean (2).csv

│   └── house\_prices\_clean.csv

├── models/

│   ├── house\_price\_model.pkl

│   └── house\_price\_preprocessor.pkl

├── notebooks/

│   └── House\_Price\_Prediction.ipynb

├── .gitignore

├── requirements.txt

└── README.md

