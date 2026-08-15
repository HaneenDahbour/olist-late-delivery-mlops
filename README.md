# Olist Late Delivery Prediction â€” MLOps Project

End-to-end MLOps project based on the **Brazilian E-Commerce Public Dataset by Olist**.

The project starts with relational database ingestion and SQL validation and will progressively develop into a machine-learning system for predicting whether an e-commerce order will be delivered late.

## Project Objective

The goal is to build a reproducible machine-learning pipeline capable of answering:

> **Will an order be delivered late or on time?**

The project is developed incrementally, beginning with data infrastructure before moving into exploratory data analysis, feature engineering, model development, deployment, and monitoring.

## Current Stage

### Task 1 â€” Relational Database Setup and Data Ingestion

Progress:

- [x] Create project workspace
- [x] Download the Olist dataset
- [x] Store the raw CSV files locally
- [x] Inspect source CSV schemas
- [x] Identify the main relational links
- [x] Configure PostgreSQL using Docker Compose
- [x] Start the PostgreSQL container
- [x] Validate PostgreSQL connectivity
- [ ] Create relational database tables
- [ ] Load CSV data into PostgreSQL
- [ ] Validate row counts and data integrity
- [ ] Run SQL queries
- [ ] Test joins between related tables
- [ ] Connect PostgreSQL to Python

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**.

The source data contains:

- Customers
- Orders
- Order items
- Products
- Sellers
- Payments
- Reviews
- Geolocation
- Product category translations

Raw CSV files are stored locally under `data/` and are intentionally excluded from Git version control.

Expected files:

```text
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
