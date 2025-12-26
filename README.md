Global Health Analytics & Machine Learning Web Application

This project is an end-to-end Global Health Analytics Web Application that transforms messy real-world health data into actionable insights through automated data cleaning, analytics, machine learning prediction, and automated reporting.

The system combines ETL pipelines, SQL analytics, machine learning, data visualization, and a deployed web interface to support data-driven decision-making for public health stakeholders.

Project Overview

The application begins with an ETL process where raw global health datasets are ingested, cleaned, standardized, and validated using pandas. This includes handling missing values, normalizing numerical features, encoding categorical variables, and ensuring data consistency for analysis and modeling.

The cleaned data is stored in a SQLite database, enabling structured, reproducible, and scalable SQL-based analysis.

Web Application Functionality

A deployed interactive web application allows users to:

Select a country and year

View automatically generated analytics based on the selected filters

Explore insights through interactive visualizations

Generate machine learning predictions

Download and receive automated reports

Analytics & Visualization

Using SQL and pandas, the application performs analytical queries to uncover trends in:

Disease prevalence

Mortality rates

Recovery rates

Treatment costs

Results are dynamically updated based on the selected country and year and displayed using interactive visualizations built with Plotly and Matplotlib, enabling clear comparison across time and regions.

Machine Learning & Prediction

The application integrates machine learning models to predict key health indicators such as mortality rates based on historical health data.

Key features include:

Data preprocessing and feature encoding

Regression-based prediction models

Feature importance analysis to explain model behavior

Model evaluation metrics to assess predictive performance

Predictions are presented directly within the web application alongside analytical insights.

Automated Reporting & Email Delivery

The system automatically generates professionally formatted PDF reports summarizing:

Country- and year-specific analytics

Visualizations and key metrics

Machine learning predictions and feature importance

Reports can be:

Downloaded directly from the web application

Automatically sent via email to stakeholders using SMTP

This makes the application suitable for real-world reporting, monitoring, and decision-support workflows.

Key Technologies

Python (pandas, NumPy)

SQLite & SQL

Machine Learning (scikit-learn)

Data Visualization (Plotly)

Web Application Framework (e.g., Streamlit)

Automated Reporting (PDF)

Email Automation (SMTP)

Project Highlights

End-to-end ETL and analytics pipeline

Interactive, deployed web application

Country- and year-based analytics

Machine learning prediction with explainability

Automated PDF report generation

Email delivery to stakeholders

Production-style architecture and workflow
