📊 Sales Performance Dashboard
https://customercommentclassifier.streamlit.app

#### *Python • MySQL • SQLAlchemy • Pandas • Tableau*

An end-to-end Business Intelligence and Data Analytics project where I built a relational database, extracted and transformed data using Python, computed sales KPIs, and visualized insights through an interactive Tableau dashboard.

---

## 🚀 Project Overview

This project simulates a real-world sales analytics workflow and covers:

* **Database creation** using MySQL
* **Synthetic data generation** for customers, products, and transactions
* **ETL (Extract, Transform, Load)** using Python + SQLAlchemy
* **Data cleaning & feature engineering**
* **KPI calculation** for sales insights
* **Interactive dashboard development** in Tableau

It demonstrates strong **data analysis**, **SQL**, **Python**, and **dashboarding** skills used in data analytics and BI roles.

---

## 🧱 Database Structure

The MySQL database consists of **three relational tables**:

| Table         | Description                                             |
| ------------- | ------------------------------------------------------- |
| **customers** | Customer information (name, region)                     |
| **products**  | Product details (name, category, price)                 |
| **sales**     | Individual transactions, linked to customers + products |

These tables are connected via **foreign keys**.

---

## 🔍 Data Processing & Transformation (Python)

Using **Python, Pandas, and SQLAlchemy**, the following steps were performed:

### ✔ Data Extraction

* Connected securely to MySQL using SQLAlchemy
* Joined customers, products, and sales tables

### ✔ Data Cleaning

* Removed records with missing or zero `sale_amount`
* Converted `sale_date` to a datetime object

### ✔ Feature Engineering

* Extracted **year** and **month** for time-series analysis

### ✔ KPI Calculations

* **Total sales by product category**
* **Total & average sale amount by region**
* **Monthly sales trend**

All results are displayed in tabular format inside the notebook.

---

## 📂 Files Included

| File                              | Description                   |
| --------------------------------- | ----------------------------- |
| **sales_dashboard.ipynb**         | Main analysis notebook        |
| **sample_customers.csv**          | Exported customer data        |
| **sample_products.csv**           | Exported product data         |
| **sample_sales.csv**              | Exported sales data           |
| **tableau_dashboard.png**         | Snapshot of Tableau dashboard |
| **sales_dashboard.py** (optional) | Script version of notebook    |
| **README.md**                     | Project documentation         |

---


## 📌 Author

**Manvendra Ray**
Data Analyst & Quantitative Finance Enthusiast
📧 Contact: *(optional)*
🔗 LinkedIn: *(optional)*


