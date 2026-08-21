-- Olist Late Delivery MLOps Project
-- Bulk-load the raw Kaggle CSV files into PostgreSQL.
--
-- Parent tables are loaded before child tables so foreign-key
-- constraints remain valid throughout ingestion.

BEGIN;

COPY customers
FROM '/data/olist_customers_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY sellers
FROM '/data/olist_sellers_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY products
FROM '/data/olist_products_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY product_category_translation
FROM '/data/product_category_name_translation.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY orders
FROM '/data/olist_orders_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY order_items
FROM '/data/olist_order_items_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY order_payments
FROM '/data/olist_order_payments_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY order_reviews
FROM '/data/olist_order_reviews_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COPY geolocation
FROM '/data/olist_geolocation_dataset.csv'
WITH (FORMAT CSV, HEADER TRUE);

COMMIT;
