# Olist Late Delivery Prediction - MLOps Project

End-to-end MLOps project based on the **Brazilian E-Commerce Public Dataset by Olist**.

The project starts with relational database ingestion and SQL validation and will progressively develop into a machine-learning system for predicting whether an e-commerce order will be delivered late.

## Project Objective

The goal is to build a reproducible machine-learning pipeline capable of answering:

> **Will an order be delivered late or on time?**

The project is developed incrementally, beginning with data infrastructure before moving into exploratory data analysis, feature engineering, model development, deployment, and monitoring.

## Current Stage

### Task 1 - Relational Database Setup and Data Ingestion

Progress:

- [x] Create project workspace
- [x] Download the Olist dataset
- [x] Store the raw CSV files locally
- [x] Inspect source CSV schemas
- [x] Identify the main relational links
- [x] Configure PostgreSQL using Docker Compose
- [x] Start the PostgreSQL container
- [x] Validate PostgreSQL connectivity
- [x] Create relational database tables
- [x] Load CSV data into PostgreSQL
- [x] Validate row counts and data integrity
- [ ] Run SQL queries
- [x] Test joins between related tables
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

## Local Python Environment

The project uses a dedicated Python virtual environment to isolate project dependencies from the system Python installation.

Python version:

```text
Python 3.14.5
```

Create the environment:

```powershell
py -3.14 -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install project dependencies:

```powershell
pip install -r requirements.txt
```

The `.venv/` directory is excluded from Git. Only `requirements.txt` is version-controlled so the Python environment can be reproduced on another machine.

## Database Architecture

The Olist dataset is delivered as raw CSV files, but the source data is relational. PostgreSQL is used as the structured data layer before exploratory analysis and machine-learning feature engineering.

Current data flow:

```text
Olist Kaggle CSV files
        |
        v
Raw data profiling
        |
        v
PostgreSQL 16 in Docker
        |
        v
Relational Olist tables
        |
        v
Data validation
        |
        v
SQL / EDA
        |
        v
Feature engineering
        |
        v
ML training pipeline
```

PostgreSQL runs inside Docker so the database environment can be recreated consistently across machines.

## Database Schema

The relational schema is defined in `sql/create_tables.sql`.

Nine source tables are created:

- `customers`
- `orders`
- `order_items`
- `order_payments`
- `order_reviews`
- `products`
- `sellers`
- `geolocation`
- `product_category_translation`

### Schema Decisions From Data Profiling

- `customer_id` is unique across 99,441 customer rows.
- `order_id` is unique across 99,441 order rows.
- `(order_id, order_item_id)` is used as the composite primary key for order items.
- `(order_id, payment_sequential)` is used as the composite primary key for payments.
- `review_id` alone is not unique; 814 duplicated values were detected.
- `(review_id, order_id)` is unique and is used as the composite primary key for reviews.
- Delivery timestamps containing legitimate source missing values remain nullable.
- The geolocation dataset contains 261,831 exact duplicate rows, so no artificial raw primary key is imposed.
- Two product category values are absent from the translation file, so that relationship is not enforced with a foreign key.
- ZIP-code prefixes are stored as character data to preserve leading zeros.

## Data Ingestion

Bulk ingestion is defined in `sql/load_data.sql`.

PostgreSQL `COPY` is used instead of row-by-row inserts for efficient loading. The load runs inside a transaction using `BEGIN` and `COMMIT`, and parent tables are loaded before child tables so foreign-key constraints remain valid.

## Data Validation

Post-ingestion validation is defined in `sql/validate_data.sql`.

| Table | Rows |
| --- | ---: |
| customers | 99,441 |
| geolocation | 1,000,163 |
| order_items | 112,650 |
| order_payments | 103,886 |
| order_reviews | 99,224 |
| orders | 99,441 |
| product_category_translation | 71 |
| products | 32,951 |
| sellers | 3,095 |

The PostgreSQL row counts match the original CSV profiling results.

The orders-to-customers relationship check also confirmed that all 99,441 orders reference an existing customer.

## Reproducing the Database

The running PostgreSQL database itself is not committed to Git. Instead, the repository stores the configuration and scripts required to recreate it.

```text
1. Obtain the Olist CSV dataset
2. Place the CSV files under data/
3. Start PostgreSQL with Docker Compose
4. Apply sql/create_tables.sql
5. Run sql/load_data.sql
6. Run sql/validate_data.sql
```

## Next Stage

The next milestone is SQL exploratory analysis and Python-to-PostgreSQL integration, followed by construction of the machine-learning training dataset.
