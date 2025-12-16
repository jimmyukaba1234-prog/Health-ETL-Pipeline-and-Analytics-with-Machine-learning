# Global Health ETL, Analytics, Machine learning Automated Reporting Pipeline

This project implements an end-to-end health data analytics pipeline that transforms raw global health data into actionable insights and automatically delivers weekly reports.

The workflow begins with an ETL process where raw health data is loaded, cleaned, standardized, and validated using pandas. Key steps include handling missing values, normalizing numerical features, and preparing the dataset for both analytical queries and machine learning. The cleaned dataset is then persisted to a SQLite database, enabling structured SQL-based analysis and reproducibility.

Using SQL and pandas, the project performs exploratory and statistical analysis to uncover trends in disease prevalence, mortality rates, recovery rates, and treatment costs across regions and time. Aggregated metrics and rankings are generated to highlight high-risk diseases and key public health indicators.

For modeling, the pipeline applies machine learning techniques (including feature encoding and regression-based prediction) to estimate mortality rates based on historical health indicators. Model evaluation metrics are computed to assess predictive performance.

Insights are visualized using interactive and static charts (Plotly and Matplotlib), enabling clear comparison across diseases and years. These visualizations are integrated into an automated reporting system.

The final stage generates a professionally formatted PDF and Excel report summarizing insights, predictions, and key metrics. Reports are automatically distributed via email using SMTP, making the pipeline suitable for real-world reporting and decision-support use cases.

Overall, this project demonstrates practical skills in data engineering, SQL analytics, machine learning, visualization, and automation, reflecting a production-style approach to health data analysis and reporting.
