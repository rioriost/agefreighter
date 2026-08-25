package meta

const schemaVersion = 7

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

var migrations = [][]string{
	migrationV1,
	migrationV2,
	migrationV3,
	migrationV4,
	migrationV5,
	migrationV6,
	migrationV7,
}
