CREATE OR REPLACE FUNCTION load_graph_from_azure_storage()
RETURNS VOID AS $$
DECLARE
    chunk RECORD;
    chunk_size BIGINT := 100000;
    num_offset BIGINT := 0;
    total_rows BIGINT;

    ENTRY_ID_BITS INTEGER := 32 + 16;
    ENTRY_ID_MASK BIGINT := 0x0000FFFFFFFFFFFF;
    oid BIGINT;
    first_id_actor BIGINT;
    first_id_film BIGINT;

    start_time TIMESTAMP;
    end_time TIMESTAMP;
    duration INTERVAL;
BEGIN
    SET search_path = ag_catalog, "$user", public;

    -- create a temporary table to store the data from the Azure Storage
    CREATE TEMP TABLE temp_from_azure_storage (
        Actor TEXT,
        ActorID TEXT,
        Film TEXT,
        Year TEXT,
        Votes TEXT,
        Rating TEXT,
        FilmID TEXT
    );

    -- bulk load from the Azure Storage into the temporary table
    INSERT INTO temp_from_azure_storage
    SELECT *
    FROM azure_storage.blob_get(
        'saagefreighter6894b1e8',
        'bcagefreighter6894b1e8',
        'actorfilms.csv',
        options := azure_storage.options_csv_get(header := 'true'))
    AS res (Actor TEXT,ActorID TEXT,Film TEXT,Year TEXT,Votes TEXT,Rating TEXT,FilmID TEXT);

    -- create a temporary table to store the mapping between the entryID and the id
    CREATE TEMP TABLE temp_id_map (entryID TEXT, id BIGINT);

    -- create the graph and vertex labels, edge label
    PERFORM create_graph('AgeFreighter');
    PERFORM create_vlabel('AgeFreighter', 'Actor');
    PERFORM create_vlabel('AgeFreighter', 'Film');
    PERFORM create_elabel('AgeFreighter', 'ACTED_IN');

    SELECT COUNT(*) INTO total_rows FROM temp_from_azure_storage;
    RAISE NOTICE 'total_rows: %', total_rows;

    -- determine the first id for the Actor vertex
    SELECT id INTO oid FROM ag_label WHERE name='Actor';
    first_id_actor := ((oid << ENTRY_ID_BITS) | (1 & ENTRY_ID_MASK));
    RAISE NOTICE 'first_id_actor: %', first_id_actor;

    -- determine the first id for the Film vertex
    SELECT id INTO oid FROM ag_label WHERE name='Film';
    first_id_film := ((oid << ENTRY_ID_BITS) | (1 & ENTRY_ID_MASK));
    RAISE NOTICE 'first_id_film: %', first_id_film;

    WHILE num_offset < total_rows LOOP
        -- bulk insert the Actor data
        RAISE NOTICE 'Inserting Actor';
        start_time := clock_timestamp();

        INSERT INTO "AgeFreighter"."Actor" (properties)
        SELECT format('{"id":"%s", "Actor":"%s"}', ActorID, Actor)::agtype
        FROM (
            SELECT DISTINCT ActorID, Actor
            FROM temp_from_azure_storage
            OFFSET num_offset LIMIT chunk_size
        ) AS distinct_actors;

        -- bulk insert the mapping between the entryID and the id
        INSERT INTO temp_id_map (entryID, id)
        SELECT distinct_actors.ActorID, first_id_actor + ROW_NUMBER() OVER () - 1
        FROM (
            SELECT DISTINCT ActorID
            FROM temp_from_azure_storage
            OFFSET num_offset LIMIT chunk_size
        ) AS distinct_actors;

        end_time := clock_timestamp();
        duration := end_time - start_time;
        RAISE NOTICE 'Time to bulk-load Actor: %', duration;

        -- bulk insert the Film data
        RAISE NOTICE 'Inserting Film';
        start_time := clock_timestamp();

        INSERT INTO "AgeFreighter"."Film" (properties)
        SELECT format('{"id":"%s", "Film":"%s", "Year":"%s", "Votes":"%s", "Rating":"%s"}', FilmID, Film, Year, Votes, Rating)::agtype
        FROM (
            SELECT DISTINCT FilmID, Film, Year, Votes, Rating
            FROM temp_from_azure_storage
            OFFSET num_offset LIMIT chunk_size
        ) AS distinct_films;

        -- bulk insert the mapping between the entryID and the id
        INSERT INTO temp_id_map (entryID, id)
        SELECT distinct_films.FilmID, first_id_film + ROW_NUMBER() OVER () - 1
        FROM (
            SELECT DISTINCT FilmID
            FROM temp_from_azure_storage
            OFFSET num_offset LIMIT chunk_size
        ) AS distinct_films;

        end_time := clock_timestamp();
        duration := end_time - start_time;
        RAISE NOTICE 'Time to bulk-load Film: %', duration;

        -- bulk insert the edge data
        RAISE NOTICE 'Inserting ACTED_IN';
        start_time := clock_timestamp();

        INSERT INTO "AgeFreighter"."ACTED_IN" (start_id, end_id)
        SELECT actor_map.id::agtype::graphid, film_map.id::agtype::graphid
        FROM (
            SELECT DISTINCT ActorID, FilmID
            FROM temp_from_azure_storage
            OFFSET num_offset LIMIT chunk_size
        ) AS af
        JOIN temp_id_map AS actor_map ON af.ActorID = actor_map.entryID
        JOIN temp_id_map AS film_map ON af.FilmID = film_map.entryID;

        end_time := clock_timestamp();
        duration := end_time - start_time;
        RAISE NOTICE 'Time to bulk-load ACTED_IN: %', duration;

        num_offset := num_offset + chunk_size;
    END LOOP;

    CREATE INDEX ON "AgeFreighter"."Actor" USING GIN (properties);
    CREATE INDEX ON "AgeFreighter"."Actor" USING BTREE (id);

    CREATE INDEX ON "AgeFreighter"."Film" USING GIN (properties);
    CREATE INDEX ON "AgeFreighter"."Film" USING BTREE (id);

    CREATE INDEX ON "AgeFreighter"."ACTED_IN" USING BTREE (start_id);
    CREATE INDEX ON "AgeFreighter"."ACTED_IN" USING BTREE (end_id);

END;
$$ LANGUAGE plpgsql;
