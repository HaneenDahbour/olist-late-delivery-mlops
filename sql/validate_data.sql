-- Olist Late Delivery MLOps Project
-- Post-ingestion validation checks.

SELECT 'customers' AS table_name, COUNT(*) AS row_count
FROM customers

UNION ALL

SELECT 'sellers', COUNT(*)
FROM sellers

UNION ALL

SELECT 'products', COUNT(*)
FROM products

UNION ALL

SELECT 'product_category_translation', COUNT(*)
FROM product_category_translation

UNION ALL

SELECT 'orders', COUNT(*)
FROM orders

UNION ALL

SELECT 'order_items', COUNT(*)
FROM order_items

UNION ALL

SELECT 'order_payments', COUNT(*)
FROM order_payments

UNION ALL

SELECT 'order_reviews', COUNT(*)
FROM order_reviews

UNION ALL

SELECT 'geolocation', COUNT(*)
FROM geolocation

ORDER BY table_name;
