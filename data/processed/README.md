# Processed Data Artifacts

Files in this directory are generated locally by the numbered notebooks and are not committed to Git.

## `orders_joined.parquet`

Created by `notebooks/01_read_and_join.ipynb`.

Verified checkpoint:

- Rows: 99,441
- Columns: 65
- Duplicate order IDs: 0
- Size: 21,975,102 bytes
- SHA-256: `EA4FA493FBE3C712CA356CCC2007F3EBA8B044AB1E7B0ADA65A974A3546B4A27`

To reproduce the artifact, start PostgreSQL through Docker Compose and execute Notebook 1 from a clean kernel.
