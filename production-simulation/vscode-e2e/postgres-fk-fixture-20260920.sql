-- Approved isolated fixture only; never run against a different database.
-- Invoke with psql -X -v ON_ERROR_STOP=1 through a protected authenticated channel.
BEGIN;
SET LOCAL statement_timeout = '20s';
SET LOCAL lock_timeout = '2s';
DO $guard$
BEGIN
  IF current_database() <> 'p1source' OR current_user <> 'afsourceadmin' THEN
    RAISE EXCEPTION 'Wrong fixture database or administrator';
  END IF;
  IF EXISTS (SELECT FROM pg_namespace WHERE nspname = 'af_fk_qualification_20260920') THEN
    RAISE EXCEPTION 'Fixture schema exists; preserve it and stop';
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agefreighter_reader') THEN
    RAISE EXCEPTION 'Expected existing reader';
  END IF;
END
$guard$;
CREATE SCHEMA af_fk_qualification_20260920;
CREATE TABLE af_fk_qualification_20260920.suppliers (
  id bigint PRIMARY KEY,
  name text NOT NULL
);
CREATE TABLE af_fk_qualification_20260920.products (
  id bigint PRIMARY KEY,
  name text NOT NULL,
  supplier_id bigint NOT NULL,
  optional_supplier_id bigint,
  CONSTRAINT products_supplier_fk FOREIGN KEY (supplier_id)
    REFERENCES af_fk_qualification_20260920.suppliers(id),
  CONSTRAINT products_optional_supplier_fk FOREIGN KEY (optional_supplier_id)
    REFERENCES af_fk_qualification_20260920.suppliers(id)
);
INSERT INTO af_fk_qualification_20260920.suppliers VALUES
  (1, 'Synthetic supplier A'), (2, 'Synthetic supplier B');
INSERT INTO af_fk_qualification_20260920.products VALUES
  (101, 'Synthetic product A', 1, NULL),
  (102, 'Synthetic product B', 1, 2),
  (103, 'Synthetic product C', 2, NULL);
GRANT USAGE ON SCHEMA af_fk_qualification_20260920 TO agefreighter_reader;
GRANT SELECT ON af_fk_qualification_20260920.suppliers,
  af_fk_qualification_20260920.products TO agefreighter_reader;
DO $verify$
BEGIN
  IF (SELECT count(*) FROM af_fk_qualification_20260920.suppliers) <> 2
     OR (SELECT count(*) FROM af_fk_qualification_20260920.products) <> 3
     OR NOT has_schema_privilege('agefreighter_reader', 'af_fk_qualification_20260920', 'USAGE')
     OR NOT has_table_privilege('agefreighter_reader', 'af_fk_qualification_20260920.suppliers', 'SELECT')
     OR NOT has_table_privilege('agefreighter_reader', 'af_fk_qualification_20260920.products', 'SELECT') THEN
    RAISE EXCEPTION 'Fixture postcondition failed';
  END IF;
END
$verify$;
COMMIT;
