package meta

const schemaVersion = 21

var migrationV1 = []string{
	`CREATE TABLE agefreighter_meta.load_job (
		job_id uuid PRIMARY KEY,
		name text NOT NULL CHECK (name <> ''),
		source_type text NOT NULL CHECK (
			source_type IN ('csv', 'postgresql', 'neo4j', 'cosmos-nosql')
		),
		load_mode text NOT NULL CHECK (
			load_mode IN ('create', 'replace', 'append', 'upsert')
		),
		target_graph text NOT NULL CHECK (target_graph <> ''),
		config_fingerprint character(64) NOT NULL CHECK (
			config_fingerprint ~ '^[0-9a-f]{64}$'
		),
		status text NOT NULL CHECK (
			status IN ('pending', 'running', 'committed', 'failed')
		),
		graph_generation_id bigint,
		next_batch_id bigint NOT NULL DEFAULT 1 CHECK (next_batch_id > 0),
		resume_token text NOT NULL DEFAULT '',
		committed_rows bigint NOT NULL DEFAULT 0 CHECK (committed_rows >= 0),
		committed_bytes bigint NOT NULL DEFAULT 0 CHECK (committed_bytes >= 0),
		rejected_rows bigint NOT NULL DEFAULT 0 CHECK (rejected_rows >= 0),
		error_message text NOT NULL DEFAULT '',
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		started_at timestamp with time zone,
		updated_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		completed_at timestamp with time zone
	)`,
	`CREATE TABLE agefreighter_meta.graph_generation (
		graph_generation_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
		job_id uuid NOT NULL UNIQUE REFERENCES agefreighter_meta.load_job(job_id),
		graph_name text NOT NULL CHECK (graph_name <> ''),
		graph_oid oid NOT NULL,
		namespace_oid oid NOT NULL,
		generation bigint NOT NULL CHECK (generation > 0),
		state text NOT NULL CHECK (state IN ('loading', 'active', 'retired')),
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		updated_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		UNIQUE (graph_oid, generation),
		CHECK (graph_oid = namespace_oid)
	)`,
	`CREATE UNIQUE INDEX graph_generation_current_name_uq
		ON agefreighter_meta.graph_generation (graph_name)
		WHERE state IN ('loading', 'active')`,
	`ALTER TABLE agefreighter_meta.load_job
		ADD CONSTRAINT load_job_graph_generation_fk
		FOREIGN KEY (graph_generation_id)
		REFERENCES agefreighter_meta.graph_generation(graph_generation_id)`,
	`CREATE TABLE agefreighter_meta.label_generation (
		label_generation_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
		graph_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.graph_generation(graph_generation_id)
			ON DELETE CASCADE,
		label_name text NOT NULL CHECK (label_name <> ''),
		kind character(1) NOT NULL CHECK (kind IN ('v', 'e')),
		graph_namespace_oid oid NOT NULL,
		label_id integer NOT NULL CHECK (label_id BETWEEN 1 AND 65535),
		relation_oid oid NOT NULL,
		sequence_oid oid NOT NULL,
		mapping_generation bigint NOT NULL CHECK (mapping_generation > 0),
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		updated_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		UNIQUE (graph_generation_id, label_name, mapping_generation),
		UNIQUE (graph_generation_id, label_id, mapping_generation)
	)`,
	`CREATE INDEX label_generation_catalog_idx
		ON agefreighter_meta.label_generation (
			graph_namespace_oid, label_id, relation_oid, mapping_generation
		)`,
	`CREATE TABLE agefreighter_meta.vertex_identity (
		graph_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.graph_generation(graph_generation_id)
			ON DELETE CASCADE,
		label_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.label_generation(label_generation_id)
			ON DELETE CASCADE,
		graph_namespace_oid oid NOT NULL,
		label_id integer NOT NULL CHECK (label_id BETWEEN 1 AND 65535),
		label_relation_oid oid NOT NULL,
		mapping_generation bigint NOT NULL CHECK (mapping_generation > 0),
		source_namespace text NOT NULL CHECK (source_namespace <> ''),
		external_id text NOT NULL CHECK (external_id <> ''),
		graph_id bigint NOT NULL,
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		PRIMARY KEY (
			graph_generation_id, label_generation_id,
			source_namespace, external_id
		),
		UNIQUE (graph_generation_id, graph_id)
	)`,
	`CREATE INDEX vertex_identity_endpoint_idx
		ON agefreighter_meta.vertex_identity (
			graph_generation_id, source_namespace, label_id, external_id
		) INCLUDE (graph_id, label_generation_id)`,
	`CREATE TABLE agefreighter_meta.edge_identity (
		graph_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.graph_generation(graph_generation_id)
			ON DELETE CASCADE,
		label_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.label_generation(label_generation_id)
			ON DELETE CASCADE,
		graph_namespace_oid oid NOT NULL,
		label_id integer NOT NULL CHECK (label_id BETWEEN 1 AND 65535),
		label_relation_oid oid NOT NULL,
		mapping_generation bigint NOT NULL CHECK (mapping_generation > 0),
		source_namespace text NOT NULL CHECK (source_namespace <> ''),
		external_id text NOT NULL CHECK (external_id <> ''),
		graph_id bigint NOT NULL,
		start_graph_id bigint NOT NULL,
		end_graph_id bigint NOT NULL,
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		PRIMARY KEY (
			graph_generation_id, label_generation_id,
			source_namespace, external_id
		),
		UNIQUE (graph_generation_id, graph_id)
	)`,
	`CREATE TABLE agefreighter_meta.load_batch (
		job_id uuid NOT NULL REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		batch_id bigint NOT NULL CHECK (batch_id > 0),
		attempt integer NOT NULL CHECK (attempt > 0),
		status text NOT NULL CHECK (status IN ('running', 'committed', 'failed')),
		rows bigint NOT NULL CHECK (rows >= 0),
		bytes bigint NOT NULL CHECK (bytes >= 0),
		first_resource text NOT NULL DEFAULT '',
		first_line bigint NOT NULL DEFAULT 0 CHECK (first_line >= 0),
		first_byte_offset bigint NOT NULL DEFAULT 0 CHECK (first_byte_offset >= 0),
		first_token text NOT NULL DEFAULT '',
		last_resource text NOT NULL DEFAULT '',
		last_line bigint NOT NULL DEFAULT 0 CHECK (last_line >= 0),
		last_byte_offset bigint NOT NULL DEFAULT 0 CHECK (last_byte_offset >= 0),
		last_token text NOT NULL DEFAULT '',
		error_message text NOT NULL DEFAULT '',
		started_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		finished_at timestamp with time zone,
		PRIMARY KEY (job_id, batch_id, attempt)
	)`,
	`CREATE UNIQUE INDEX load_batch_committed_uq
		ON agefreighter_meta.load_batch (job_id, batch_id)
		WHERE status = 'committed'`,
	`CREATE INDEX load_batch_latest_idx
		ON agefreighter_meta.load_batch (job_id, batch_id DESC, attempt DESC)`,
	`CREATE TABLE agefreighter_meta.reject_record (
		job_id uuid NOT NULL,
		batch_id bigint NOT NULL,
		attempt integer NOT NULL,
		resume_token text NOT NULL CHECK (resume_token <> ''),
		resource text NOT NULL DEFAULT '',
		line bigint NOT NULL DEFAULT 0 CHECK (line >= 0),
		byte_offset bigint NOT NULL DEFAULT 0 CHECK (byte_offset >= 0),
		error_class text NOT NULL CHECK (error_class <> ''),
		error_message text NOT NULL CHECK (error_message <> ''),
		record jsonb,
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		PRIMARY KEY (job_id, batch_id, attempt, resume_token),
		FOREIGN KEY (job_id, batch_id, attempt)
			REFERENCES agefreighter_meta.load_batch(job_id, batch_id, attempt)
			ON DELETE CASCADE
	)`,
	`CREATE INDEX reject_record_attempt_idx
		ON agefreighter_meta.reject_record (job_id, batch_id, attempt)`,
}

var migrationV2 = []string{
	`ALTER TABLE agefreighter_meta.load_batch
		ADD COLUMN rejected_rows bigint NOT NULL DEFAULT 0
		CHECK (rejected_rows >= 0)`,
}

var migrationV3 = []string{
	`ALTER TABLE agefreighter_meta.label_generation
		ADD CONSTRAINT label_generation_identity_key UNIQUE (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, relation_oid, mapping_generation
		)`,
	`ALTER TABLE agefreighter_meta.vertex_identity
		DROP CONSTRAINT vertex_identity_label_generation_id_fkey,
		ADD CONSTRAINT vertex_identity_label_generation_fk
		FOREIGN KEY (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, label_relation_oid, mapping_generation
		)
		REFERENCES agefreighter_meta.label_generation (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, relation_oid, mapping_generation
		)
		ON DELETE CASCADE`,
	`ALTER TABLE agefreighter_meta.edge_identity
		DROP CONSTRAINT edge_identity_label_generation_id_fkey,
		ADD CONSTRAINT edge_identity_label_generation_fk
		FOREIGN KEY (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, label_relation_oid, mapping_generation
		)
		REFERENCES agefreighter_meta.label_generation (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, relation_oid, mapping_generation
		)
		ON DELETE CASCADE`,
}

var migrationV4 = []string{
	`CREATE UNIQUE INDEX load_batch_running_uq
		ON agefreighter_meta.load_batch (job_id, batch_id)
		WHERE status = 'running'`,
}

var migrationV5 = []string{
	`ALTER TABLE agefreighter_meta.label_generation
		ADD CONSTRAINT label_generation_identity_kind_key UNIQUE (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, relation_oid, mapping_generation, kind
		)`,
	`ALTER TABLE agefreighter_meta.vertex_identity
		ADD COLUMN label_kind character(1) NOT NULL DEFAULT 'v'
			CHECK (label_kind = 'v')`,
	`ALTER TABLE agefreighter_meta.vertex_identity
		ALTER COLUMN label_kind DROP DEFAULT`,
	`ALTER TABLE agefreighter_meta.vertex_identity
		DROP CONSTRAINT vertex_identity_label_generation_fk,
		ADD CONSTRAINT vertex_identity_label_generation_fk
		FOREIGN KEY (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, label_relation_oid, mapping_generation, label_kind
		)
		REFERENCES agefreighter_meta.label_generation (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, relation_oid, mapping_generation, kind
		)
		ON DELETE CASCADE`,
	`ALTER TABLE agefreighter_meta.edge_identity
		ADD COLUMN label_kind character(1) NOT NULL DEFAULT 'e'
			CHECK (label_kind = 'e')`,
	`ALTER TABLE agefreighter_meta.edge_identity
		ALTER COLUMN label_kind DROP DEFAULT`,
	`ALTER TABLE agefreighter_meta.edge_identity
		DROP CONSTRAINT edge_identity_label_generation_fk,
		ADD CONSTRAINT edge_identity_label_generation_fk
		FOREIGN KEY (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, label_relation_oid, mapping_generation, label_kind
		)
		REFERENCES agefreighter_meta.label_generation (
			label_generation_id, graph_generation_id, graph_namespace_oid,
			label_id, relation_oid, mapping_generation, kind
		)
		ON DELETE CASCADE`,
}

var migrationV6 = []string{
	`LOCK TABLE
		agefreighter_meta.vertex_identity,
		agefreighter_meta.edge_identity
		IN SHARE MODE`,
	`DO $$
	BEGIN
		IF EXISTS (
			SELECT 1
			FROM agefreighter_meta.vertex_identity
			GROUP BY graph_generation_id, source_namespace, label_id, external_id
			HAVING COUNT(*) > 1
		) THEN
			RAISE EXCEPTION
				'cannot enforce vertex identity lookup uniqueness: conflicting mapping generations exist'
				USING ERRCODE = '23505';
		END IF;
	END
	$$`,
	`DO $$
	BEGIN
		IF EXISTS (
			SELECT 1
			FROM agefreighter_meta.edge_identity
			GROUP BY graph_generation_id, source_namespace, label_id, external_id
			HAVING COUNT(*) > 1
		) THEN
			RAISE EXCEPTION
				'cannot enforce edge identity lookup uniqueness: conflicting mapping generations exist'
				USING ERRCODE = '23505';
		END IF;
	END
	$$`,
	`CREATE UNIQUE INDEX vertex_identity_lookup_uq
		ON agefreighter_meta.vertex_identity (
			graph_generation_id, source_namespace, label_id, external_id
		)`,
	`CREATE UNIQUE INDEX edge_identity_lookup_uq
		ON agefreighter_meta.edge_identity (
			graph_generation_id, source_namespace, label_id, external_id
		)`,
}

var migrationV7 = []string{
	`ALTER TABLE agefreighter_meta.vertex_identity
			ADD CONSTRAINT vertex_identity_graph_id_check CHECK (
				((graph_id >> 48) & 65535) = label_id
				AND (graph_id & 281474976710655) <> 0
			)`,
	`ALTER TABLE agefreighter_meta.edge_identity
			ADD CONSTRAINT edge_identity_graph_id_check CHECK (
				((graph_id >> 48) & 65535) = label_id
				AND (graph_id & 281474976710655) <> 0
			),
			ADD CONSTRAINT edge_identity_start_graph_id_check CHECK (
				((start_graph_id >> 48) & 65535) <> 0
				AND (start_graph_id & 281474976710655) <> 0
			),
			ADD CONSTRAINT edge_identity_end_graph_id_check CHECK (
				((end_graph_id >> 48) & 65535) <> 0
				AND (end_graph_id & 281474976710655) <> 0
			)`,
}

var migrationV8 = []string{
	`ALTER TABLE agefreighter_meta.load_job
		ADD COLUMN source_rejected_rows bigint NOT NULL DEFAULT 0
		CHECK (source_rejected_rows >= 0 AND source_rejected_rows <= rejected_rows)`,
}

var migrationV9 = []string{
	`LOCK TABLE
		agefreighter_meta.vertex_identity,
		agefreighter_meta.edge_identity
		IN SHARE ROW EXCLUSIVE MODE`,
	`ALTER TABLE agefreighter_meta.vertex_identity
		DROP CONSTRAINT IF EXISTS vertex_identity_pkey,
		DROP CONSTRAINT IF EXISTS vertex_identity_graph_generation_id_graph_id_key,
		DROP CONSTRAINT IF EXISTS vertex_identity_graph_generation_id_fkey,
		DROP CONSTRAINT IF EXISTS vertex_identity_label_generation_fk`,
	`ALTER TABLE agefreighter_meta.edge_identity
		DROP CONSTRAINT IF EXISTS edge_identity_pkey,
		DROP CONSTRAINT IF EXISTS edge_identity_graph_generation_id_graph_id_key,
		DROP CONSTRAINT IF EXISTS edge_identity_graph_generation_id_fkey,
		DROP CONSTRAINT IF EXISTS edge_identity_label_generation_fk`,
	`DROP INDEX IF EXISTS agefreighter_meta.vertex_identity_endpoint_idx`,
	`DROP INDEX IF EXISTS agefreighter_meta.vertex_identity_lookup_uq`,
	`CREATE UNIQUE INDEX vertex_identity_lookup_uq
		ON agefreighter_meta.vertex_identity (
			graph_generation_id, source_namespace, label_id, external_id
		) INCLUDE (graph_id, label_generation_id)`,
	`CREATE OR REPLACE FUNCTION agefreighter_meta.validate_vertex_identity_generation()
		RETURNS trigger
		LANGUAGE plpgsql
		AS $$
	BEGIN
		IF EXISTS (
			SELECT 1
			FROM inserted_identity i
			LEFT JOIN agefreighter_meta.label_generation l
			  ON l.label_generation_id = i.label_generation_id
			 AND l.graph_generation_id = i.graph_generation_id
			 AND l.graph_namespace_oid = i.graph_namespace_oid
			 AND l.label_id = i.label_id
			 AND l.relation_oid = i.label_relation_oid
			 AND l.mapping_generation = i.mapping_generation
			 AND l.kind = i.label_kind
			WHERE l.label_generation_id IS NULL
		) THEN
			RAISE foreign_key_violation
				USING MESSAGE = 'vertex identity does not match its label generation';
		END IF;
		RETURN NULL;
	END
	$$`,
	`CREATE OR REPLACE FUNCTION agefreighter_meta.validate_edge_identity_generation()
		RETURNS trigger
		LANGUAGE plpgsql
		AS $$
	BEGIN
		IF EXISTS (
			SELECT 1
			FROM inserted_identity i
			LEFT JOIN agefreighter_meta.label_generation l
			  ON l.label_generation_id = i.label_generation_id
			 AND l.graph_generation_id = i.graph_generation_id
			 AND l.graph_namespace_oid = i.graph_namespace_oid
			 AND l.label_id = i.label_id
			 AND l.relation_oid = i.label_relation_oid
			 AND l.mapping_generation = i.mapping_generation
			 AND l.kind = i.label_kind
			WHERE l.label_generation_id IS NULL
		) THEN
			RAISE foreign_key_violation
				USING MESSAGE = 'edge identity does not match its label generation';
		END IF;
		RETURN NULL;
	END
	$$`,
	`CREATE TRIGGER vertex_identity_generation_insert
		AFTER INSERT ON agefreighter_meta.vertex_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_vertex_identity_generation()`,
	`CREATE TRIGGER vertex_identity_generation_update
		AFTER UPDATE ON agefreighter_meta.vertex_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_vertex_identity_generation()`,
	`CREATE TRIGGER edge_identity_generation_insert
		AFTER INSERT ON agefreighter_meta.edge_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_edge_identity_generation()`,
	`CREATE TRIGGER edge_identity_generation_update
		AFTER UPDATE ON agefreighter_meta.edge_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_edge_identity_generation()`,
	`CREATE OR REPLACE FUNCTION agefreighter_meta.delete_label_generation_identities()
		RETURNS trigger
		LANGUAGE plpgsql
		AS $$
	BEGIN
		DELETE FROM agefreighter_meta.vertex_identity i
		USING deleted_generation d
		WHERE i.label_generation_id = d.label_generation_id;
		DELETE FROM agefreighter_meta.edge_identity i
		USING deleted_generation d
		WHERE i.label_generation_id = d.label_generation_id;
		RETURN NULL;
	END
	$$`,
	`CREATE TRIGGER label_generation_identity_delete
		AFTER DELETE ON agefreighter_meta.label_generation
		REFERENCING OLD TABLE AS deleted_generation
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.delete_label_generation_identities()`,
}

var migrationV10 = []string{
	`DROP TRIGGER IF EXISTS vertex_identity_generation_insert
		ON agefreighter_meta.vertex_identity`,
	`DROP TRIGGER IF EXISTS vertex_identity_generation_update
		ON agefreighter_meta.vertex_identity`,
	`DROP TRIGGER IF EXISTS edge_identity_generation_insert
		ON agefreighter_meta.edge_identity`,
	`DROP TRIGGER IF EXISTS edge_identity_generation_update
		ON agefreighter_meta.edge_identity`,
	`DROP FUNCTION IF EXISTS agefreighter_meta.validate_vertex_identity_generation()`,
	`DROP FUNCTION IF EXISTS agefreighter_meta.validate_edge_identity_generation()`,
	`ALTER TABLE agefreighter_meta.vertex_identity
		DROP COLUMN graph_namespace_oid,
		DROP COLUMN label_relation_oid,
		DROP COLUMN mapping_generation,
		DROP COLUMN label_kind`,
	`ALTER TABLE agefreighter_meta.edge_identity
		DROP COLUMN graph_namespace_oid,
		DROP COLUMN label_relation_oid,
		DROP COLUMN mapping_generation,
		DROP COLUMN label_kind`,
	`CREATE FUNCTION agefreighter_meta.validate_vertex_identity_generation()
		RETURNS trigger
		LANGUAGE plpgsql
		AS $$
	BEGIN
		IF EXISTS (
			SELECT 1
			FROM inserted_identity i
			LEFT JOIN agefreighter_meta.label_generation l
			  ON l.label_generation_id = i.label_generation_id
			 AND l.graph_generation_id = i.graph_generation_id
			 AND l.label_id = i.label_id
			 AND l.kind = 'v'
			WHERE l.label_generation_id IS NULL
		) THEN
			RAISE foreign_key_violation
				USING MESSAGE = 'vertex identity does not match its label generation';
		END IF;
		RETURN NULL;
	END
	$$`,
	`CREATE FUNCTION agefreighter_meta.validate_edge_identity_generation()
		RETURNS trigger
		LANGUAGE plpgsql
		AS $$
	BEGIN
		IF EXISTS (
			SELECT 1
			FROM inserted_identity i
			LEFT JOIN agefreighter_meta.label_generation l
			  ON l.label_generation_id = i.label_generation_id
			 AND l.graph_generation_id = i.graph_generation_id
			 AND l.label_id = i.label_id
			 AND l.kind = 'e'
			WHERE l.label_generation_id IS NULL
		) THEN
			RAISE foreign_key_violation
				USING MESSAGE = 'edge identity does not match its label generation';
		END IF;
		RETURN NULL;
	END
	$$`,
	`CREATE TRIGGER vertex_identity_generation_insert
		AFTER INSERT ON agefreighter_meta.vertex_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_vertex_identity_generation()`,
	`CREATE TRIGGER vertex_identity_generation_update
		AFTER UPDATE ON agefreighter_meta.vertex_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_vertex_identity_generation()`,
	`CREATE TRIGGER edge_identity_generation_insert
		AFTER INSERT ON agefreighter_meta.edge_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_edge_identity_generation()`,
	`CREATE TRIGGER edge_identity_generation_update
		AFTER UPDATE ON agefreighter_meta.edge_identity
		REFERENCING NEW TABLE AS inserted_identity
		FOR EACH STATEMENT
		EXECUTE FUNCTION agefreighter_meta.validate_edge_identity_generation()`,
}

var migrationV11 = []string{
	`DROP TRIGGER IF EXISTS vertex_identity_generation_insert
		ON agefreighter_meta.vertex_identity`,
	`DROP TRIGGER IF EXISTS edge_identity_generation_insert
		ON agefreighter_meta.edge_identity`,
}

var migrationV12 = []string{
	`ALTER TABLE agefreighter_meta.load_job
		ADD COLUMN backup_graph_name text NOT NULL DEFAULT '',
		ADD COLUMN backup_cleaned_at timestamp with time zone,
		ADD CONSTRAINT load_job_backup_mode_check CHECK (
			backup_graph_name = '' OR load_mode = 'replace'
		),
		ADD CONSTRAINT load_job_backup_cleanup_check CHECK (
			backup_cleaned_at IS NULL OR backup_graph_name <> ''
		)`,
	`ALTER TABLE agefreighter_meta.graph_generation
		ADD COLUMN replaces_graph_oid oid,
		ADD CONSTRAINT graph_generation_replaces_oid_check CHECK (
			replaces_graph_oid IS NULL OR replaces_graph_oid::bigint > 0
		)`,
}

var migrationV13 = []string{
	`ALTER TABLE agefreighter_meta.graph_generation
		DROP CONSTRAINT graph_generation_replaces_oid_check,
		ADD CONSTRAINT graph_generation_replaces_oid_check CHECK (
			replaces_graph_oid IS NULL OR (
				replaces_graph_oid::bigint > 0
				AND replaces_graph_oid <> graph_oid
			)
		)`,
	`ALTER TABLE agefreighter_meta.load_job
		ADD CONSTRAINT load_job_backup_target_check CHECK (
			backup_graph_name = '' OR backup_graph_name <> target_graph
		)`,
}

var migrationV14 = []string{
	`CREATE TABLE agefreighter_meta.deferred_edge (
		deferred_edge_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
		graph_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.graph_generation(graph_generation_id)
			ON DELETE CASCADE,
		label_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.label_generation(label_generation_id)
			ON DELETE CASCADE,
		label_id integer NOT NULL CHECK (label_id BETWEEN 1 AND 65535),
		job_id uuid NOT NULL
			REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		source_namespace text NOT NULL CHECK (source_namespace <> ''),
		external_id text,
		start_namespace text NOT NULL CHECK (start_namespace <> ''),
		start_label_id integer NOT NULL CHECK (start_label_id BETWEEN 1 AND 65535),
		start_external_id text NOT NULL CHECK (start_external_id <> ''),
		end_namespace text NOT NULL CHECK (end_namespace <> ''),
		end_label_id integer NOT NULL CHECK (end_label_id BETWEEN 1 AND 65535),
		end_external_id text NOT NULL CHECK (end_external_id <> ''),
		properties text NOT NULL CHECK (properties::jsonb IS NOT NULL),
		load_mode text NOT NULL CHECK (load_mode IN ('append', 'upsert')),
		append_duplicate text NOT NULL CHECK (
			append_duplicate IN ('error', 'ignore-identical')
		),
		property_mode text NOT NULL CHECK (
			property_mode IN ('replace', 'merge', 'merge-delete-null')
		),
		resource text NOT NULL DEFAULT '',
		line bigint NOT NULL DEFAULT 0 CHECK (line >= 0),
		byte_offset bigint NOT NULL DEFAULT 0 CHECK (byte_offset >= 0),
		resume_token text NOT NULL CHECK (resume_token <> ''),
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		UNIQUE (graph_generation_id, job_id, resume_token),
		CHECK (load_mode <> 'upsert' OR external_id IS NOT NULL)
	)`,
	`CREATE INDEX deferred_edge_resolution_idx
		ON agefreighter_meta.deferred_edge (
			graph_generation_id,
			start_namespace, start_label_id, start_external_id,
			end_namespace, end_label_id, end_external_id
		)`,
}

var migrationV15 = []string{
	`CREATE TABLE agefreighter_meta.connector_telemetry (
		job_id uuid PRIMARY KEY
			REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		connector text NOT NULL CHECK (
			connector IN ('csv', 'postgresql', 'neo4j', 'cosmos-nosql')
		),
		pages bigint NOT NULL CHECK (pages >= 0),
		request_charge double precision NOT NULL CHECK (
			request_charge >= 0
			AND request_charge <> 'Infinity'::double precision
			AND request_charge <> 'NaN'::double precision
		),
		failed_request_attempts bigint NOT NULL CHECK (
			failed_request_attempts >= 0
		),
		throttled_requests bigint NOT NULL CHECK (throttled_requests >= 0),
		continuation_digest text NOT NULL DEFAULT '' CHECK (
			octet_length(continuation_digest) <= 128
			AND (
				continuation_digest = ''
				OR continuation_digest ~ '^[0-9a-f]{8,128}$'
			)
		),
		recorded_at timestamp with time zone NOT NULL DEFAULT clock_timestamp()
	)`,
}

var migrationV16 = []string{
	`CREATE TABLE agefreighter_meta.diagnostic_history (
		diagnostic_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
		recorded_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		outcome text NOT NULL CHECK (
			outcome IN ('pass', 'fail', 'incomplete')
		),
		target_graph text NOT NULL CHECK (
			target_graph <> '' AND octet_length(target_graph) <= 63
		),
		postgresql_version_number integer NOT NULL CHECK (
			postgresql_version_number >= 0
		),
		age_version text NOT NULL CHECK (octet_length(age_version) <= 64),
		metadata_schema_version integer NOT NULL CHECK (
			metadata_schema_version >= 0
		),
		report jsonb NOT NULL CHECK (
			jsonb_typeof(report) = 'object'
			AND octet_length(report::text) <= 4194304
		)
	)`,
	`CREATE INDEX diagnostic_history_recent_idx
		ON agefreighter_meta.diagnostic_history (
			recorded_at DESC, diagnostic_id DESC
		)`,
}

var migrationV17 = []string{
	`CREATE UNIQUE INDEX vertex_identity_graph_id_uq
		ON agefreighter_meta.vertex_identity (
			graph_generation_id, graph_id
		)`,
	`CREATE INDEX vertex_identity_label_graph_id_idx
		ON agefreighter_meta.vertex_identity (
			graph_generation_id, label_generation_id, graph_id
		)`,
	`CREATE UNIQUE INDEX edge_identity_graph_id_uq
		ON agefreighter_meta.edge_identity (
			graph_generation_id, graph_id
		)`,
	`CREATE INDEX edge_identity_label_graph_id_idx
		ON agefreighter_meta.edge_identity (
			graph_generation_id, label_generation_id, graph_id
		)`,
	`CREATE TABLE agefreighter_meta.job_verification (
		job_id uuid PRIMARY KEY
			REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		submitted_config_fingerprint character(64) NOT NULL CHECK (
			submitted_config_fingerprint ~ '^[0-9a-f]{64}$'
		),
		resolved_mapping_fingerprint character(64) NOT NULL CHECK (
			resolved_mapping_fingerprint ~ '^[0-9a-f]{64}$'
		),
		resolved_mapping_summary jsonb NOT NULL CHECK (
			jsonb_typeof(resolved_mapping_summary) = 'object'
			AND octet_length(resolved_mapping_summary::text) <= 1048576
		)
	)`,
	`CREATE TABLE agefreighter_meta.job_label_counter (
		job_id uuid NOT NULL
			REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		label_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.label_generation(label_generation_id)
			ON DELETE CASCADE,
		kind character(1) NOT NULL CHECK (kind IN ('v', 'e')),
		counter_completeness text NOT NULL CHECK (
			counter_completeness IN ('complete', 'incomplete')
		),
		counter_provenance text NOT NULL CHECK (
			counter_provenance IN (
				'v17-lifecycle', 'legacy-resume', 'baseline-unavailable'
			)
		),
		accepted_rows bigint CHECK (accepted_rows >= 0),
		committed_rows bigint CHECK (committed_rows >= 0),
		committed_bytes bigint CHECK (committed_bytes >= 0),
		rejected_rows bigint CHECK (rejected_rows >= 0),
		updated_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		PRIMARY KEY (job_id, label_generation_id),
		CHECK (
			(
				counter_completeness = 'complete'
				AND accepted_rows IS NOT NULL
				AND committed_rows IS NOT NULL
				AND rejected_rows IS NOT NULL
			)
			OR
			(
				counter_completeness = 'incomplete'
				AND accepted_rows IS NULL
				AND committed_rows IS NULL
				AND committed_bytes IS NULL
				AND rejected_rows IS NULL
			)
		)
	)`,
	`CREATE INDEX job_label_counter_label_idx
		ON agefreighter_meta.job_label_counter (
			label_generation_id, job_id
		)`,
	`CREATE TABLE agefreighter_meta.load_batch_label_counter (
		job_id uuid NOT NULL,
		batch_id bigint NOT NULL,
		attempt integer NOT NULL,
		label_generation_id bigint NOT NULL
			REFERENCES agefreighter_meta.label_generation(label_generation_id)
			ON DELETE CASCADE,
		kind character(1) NOT NULL CHECK (kind IN ('v', 'e')),
		accepted_rows bigint NOT NULL CHECK (accepted_rows >= 0),
		committed_rows_delta bigint NOT NULL CHECK (committed_rows_delta >= 0),
		committed_bytes bigint CHECK (committed_bytes >= 0),
		rejected_rows bigint NOT NULL CHECK (rejected_rows >= 0),
		PRIMARY KEY (
			job_id, batch_id, attempt, label_generation_id
		),
		FOREIGN KEY (job_id, batch_id, attempt)
			REFERENCES agefreighter_meta.load_batch(job_id, batch_id, attempt)
			ON DELETE CASCADE
	)`,
	`CREATE TABLE agefreighter_meta.job_unclassified_counter (
		job_id uuid PRIMARY KEY
			REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		rejected_rows bigint NOT NULL DEFAULT 0 CHECK (rejected_rows >= 0),
		updated_at timestamp with time zone NOT NULL DEFAULT clock_timestamp()
	)`,
}

var migrationV18 = []string{
	`ALTER TABLE agefreighter_meta.load_job
		ADD COLUMN target_backend text NOT NULL DEFAULT 'apache-age',
		ADD COLUMN target_schema text NOT NULL DEFAULT ''`,
	`ALTER TABLE agefreighter_meta.load_job
		ADD CONSTRAINT load_job_target_backend_ck CHECK (
			target_backend IN ('apache-age', 'postgresql-property-graph')
		),
		ADD CONSTRAINT load_job_target_schema_ck CHECK (
			octet_length(target_schema) <= 63
		),
		ADD CONSTRAINT load_job_target_identity_ck CHECK (
			(target_backend = 'apache-age' AND target_schema = '')
			OR
			(target_backend = 'postgresql-property-graph' AND target_schema <> '')
		)`,
	`ALTER TABLE agefreighter_meta.load_job
		ALTER COLUMN target_backend DROP DEFAULT,
		ALTER COLUMN target_schema DROP DEFAULT`,
}

var migrationV19 = []string{
	`CREATE TABLE agefreighter_meta.property_graph_generation (
		job_id uuid PRIMARY KEY
			REFERENCES agefreighter_meta.load_job(job_id)
			ON DELETE CASCADE,
		target_schema text NOT NULL CHECK (
			target_schema <> '' AND octet_length(target_schema) <= 63
		),
		graph_name text NOT NULL CHECK (
			graph_name <> '' AND octet_length(graph_name) <= 63
		),
		definition_fingerprint character(64) NOT NULL CHECK (
			definition_fingerprint ~ '^[0-9a-f]{64}$'
		),
		state text NOT NULL CHECK (state IN ('loading', 'active')),
		created_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		updated_at timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
		UNIQUE (target_schema, graph_name)
	)`,
	`CREATE TABLE agefreighter_meta.property_graph_label (
		job_id uuid NOT NULL
			REFERENCES agefreighter_meta.property_graph_generation(job_id)
			ON DELETE CASCADE,
		label_name text NOT NULL CHECK (
			label_name <> '' AND octet_length(label_name) <= 63
		),
		kind character(1) NOT NULL CHECK (kind IN ('v', 'e')),
		table_name text NOT NULL CHECK (
			table_name <> '' AND octet_length(table_name) <= 63
		),
		start_label text,
		end_label text,
		PRIMARY KEY (job_id, label_name),
		UNIQUE (job_id, table_name),
		CHECK (
			(kind = 'v' AND start_label IS NULL AND end_label IS NULL)
			OR
			(kind = 'e' AND start_label IS NOT NULL AND end_label IS NOT NULL)
		)
	)`,
	`CREATE INDEX property_graph_generation_state_idx
		ON agefreighter_meta.property_graph_generation (
			state, target_schema, graph_name
		)`,
}

var migrationV20 = []string{
	`ALTER TABLE agefreighter_meta.property_graph_generation
		ADD COLUMN digest_root character(64) CHECK (
			digest_root IS NULL OR digest_root ~ '^[0-9a-f]{64}$'
		),
		ADD COLUMN digest_rows bigint CHECK (
			digest_rows IS NULL OR digest_rows >= 0
		),
		ADD COLUMN digest_range_count integer CHECK (
			digest_range_count IS NULL OR digest_range_count > 0
		)`,
	`CREATE TABLE agefreighter_meta.property_graph_digest_range (
		job_id uuid NOT NULL,
		label_name text NOT NULL,
		kind character(1) NOT NULL CHECK (kind IN ('v', 'e')),
		range_id integer NOT NULL CHECK (range_id BETWEEN 0 AND 255),
		row_count bigint NOT NULL CHECK (row_count > 0),
		digest character(64) NOT NULL CHECK (
			digest ~ '^[0-9a-f]{64}$'
		),
		PRIMARY KEY (job_id, label_name, range_id),
		FOREIGN KEY (job_id, label_name)
			REFERENCES agefreighter_meta.property_graph_label(job_id, label_name)
			ON DELETE CASCADE
	)`,
}

var migrationV21 = []string{
	`ALTER TABLE agefreighter_meta.property_graph_generation
		DROP CONSTRAINT property_graph_generation_target_schema_graph_name_key,
		DROP CONSTRAINT property_graph_generation_state_check,
		ADD CONSTRAINT property_graph_generation_state_check CHECK (
			state IN ('loading', 'active', 'superseded', 'retained-backup')
		)`,
	`CREATE UNIQUE INDEX property_graph_generation_active_target_idx
		ON agefreighter_meta.property_graph_generation (target_schema, graph_name)
		WHERE state = 'active'`,
}

var migrations = [][]string{
	migrationV1,
	migrationV2,
	migrationV3,
	migrationV4,
	migrationV5,
	migrationV6,
	migrationV7,
	migrationV8,
	migrationV9,
	migrationV10,
	migrationV11,
	migrationV12,
	migrationV13,
	migrationV14,
	migrationV15,
	migrationV16,
	migrationV17,
	migrationV18,
	migrationV19,
	migrationV20,
	migrationV21,
}
