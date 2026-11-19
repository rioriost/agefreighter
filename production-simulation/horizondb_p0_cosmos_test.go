package simulation_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	cosmossource "github.com/rioriost/agefreighter/internal/source/cosmos"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

func TestHorizonDBP0CosmosSourceIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P0_COSMOS_SOURCE") != "1" {
		t.Skip("explicit RU-bounded Cosmos P0 source preparation required")
	}
	output := os.Getenv("AGEFREIGHTER_HORIZON_COSMOS_SOURCE_OUTPUT")
	source := os.Getenv("AGEFREIGHTER_HORIZON_COSMOS_SOURCE")
	if !filepath.IsAbs(output) {
		t.Fatal("new absolute source evidence directory required")
	}
	if err := os.Mkdir(output, 0700); err != nil {
		t.Fatal(err)
	}
	paths, manifests, directories := horizonP0CosmosFixtures(t, output, source)
	ctx, cancel := context.WithTimeout(t.Context(), 40*time.Minute)
	defer cancel()
	if err := prepareHorizonP0Cosmos(t, ctx, paths, manifests, directories, source, output, true); err != nil {
		t.Fatal(err)
	}
	t.Log("Cosmos source hashes, actual counts, serialized configurations and production iterator verified; no HorizonDB migration")
}

func horizonP0CosmosContainer(source, run, dataset string) (string, error) {
	if (source != "cosmos" && source != "gremlin") ||
		!regexp.MustCompile(`^r[a-z0-9]{6,30}$`).MatchString(run) ||
		(dataset != "base" && dataset != "append" && dataset != "upsert") {
		return "", errors.New("new run-scoped immutable Cosmos P0 fixture required")
	}
	return "hs-p0-" + source + "-" + run + "-" + dataset, nil
}

func validateHorizonP0CosmosEndpoint(endpoint string) error {
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "https" || u.Hostname() != "afht20261118ignrcapglbrq2.documents.azure.com" ||
		(u.Port() != "" && u.Port() != "443") || (u.Path != "" && u.Path != "/") ||
		u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("exact retained private Cosmos fixture endpoint required")
	}
	return nil
}

func horizonP0CosmosJob(job *config.LoadJob, source, run, dataset string, records int64) error {
	container, err := horizonP0CosmosContainer(source, run, dataset)
	if err != nil {
		return err
	}
	c := job.Source.Cosmos
	if job.Source.Type != config.SourceCosmos || c == nil {
		return errors.New("explicit Cosmos P0 configuration required")
	}
	if err := validateHorizonP0CosmosEndpoint(c.Endpoint); err != nil {
		return err
	}
	want := int64(4580)
	if dataset == "base" {
		want = 458000
	}
	if records != want {
		return errors.New("approved P0 base or one-percent delta required")
	}
	c.Database, c.PageSize = "agefreighter", 1000
	if source == "gremlin" {
		if c.Gremlin == nil || !c.Gremlin.Enabled {
			return errors.New("serializable Gremlin discovery configuration required")
		}
		c.Gremlin.Container, c.Gremlin.MaxDiscoveryDocuments = container, int(records)
	} else {
		if c.Gremlin != nil || len(c.Vertices) == 0 || len(c.Edges) == 0 {
			return errors.New("explicit NoSQL vertex and edge mappings required")
		}
		for i := range c.Vertices {
			c.Vertices[i].Container = container
		}
		for i := range c.Edges {
			c.Edges[i].Container = container
		}
	}
	return nil
}

func horizonP0CosmosFixtures(t *testing.T, output, source string) (rangedigest.Scenario, map[string]portable.Manifest, map[string]string) {
	t.Helper()
	if source != "cosmos" && source != "gremlin" {
		t.Fatal("explicit NoSQL or Gremlin-document fixture required")
	}
	paths, manifests, directories := horizonP0Fixtures(t, output)
	if source == "gremlin" {
		for _, dataset := range []string{"base", "append", "upsert"} {
			path := map[string]string{"base": paths.Base, "append": paths.Append, "upsert": paths.Upsert}[dataset]
			dir := filepath.Join(output, "portable-gremlin-"+dataset)
			m, err := portable.ExportGremlin(t.Context(), path, dir)
			if err != nil {
				t.Fatal(err)
			}
			manifests[dataset], directories[dataset] = m, dir
		}
	}
	return paths, manifests, directories
}

func prepareHorizonP0Cosmos(t *testing.T, ctx context.Context, paths rangedigest.Scenario,
	manifests map[string]portable.Manifest, directories map[string]string, source, output string, seed bool,
) error {
	t.Helper()
	run := os.Getenv("AGEFREIGHTER_HORIZON_COSMOS_SOURCE_RUN_ID")
	commit := os.Getenv("AGEFREIGHTER_HORIZON_CANDIDATE_COMMIT")
	if _, err := horizonP0CosmosContainer(source, run, "base"); err != nil {
		return err
	}
	if !regexp.MustCompile(`^[a-f0-9]{40}$`).MatchString(commit) {
		return errors.New("exact Cosmos qualification candidate required")
	}
	endpoint := os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT")
	if err := validateHorizonP0CosmosEndpoint(endpoint); err != nil {
		return err
	}
	u, _ := url.Parse(endpoint)
	addresses, err := net.DefaultResolver.LookupIPAddr(ctx, u.Hostname())
	if err != nil || len(addresses) == 0 {
		return errors.New("Cosmos private endpoint did not resolve")
	}
	_, subnet, err := net.ParseCIDR("10.86.0.0/16")
	if err != nil {
		return err
	}
	for _, address := range addresses {
		if !subnet.Contains(address.IP) {
			return errors.New("Cosmos endpoint resolves outside the retained private VNet")
		}
	}
	var reused struct {
		Source, CandidateCommit, ScopeRunID string
		ProductionIteratorVerified          bool
		ConfigRoundTripVerified             bool
		SourceWritesPerformed               bool
		Datasets                            map[string]struct {
			Records           int64
			FixtureRootSHA256 string `json:"fixtureRootSha256"`
		}
	}
	if !seed {
		body, err := os.ReadFile(os.Getenv("AGEFREIGHTER_HORIZON_COSMOS_SOURCE_RECEIPT"))
		if err != nil {
			return err
		}
		if err := json.Unmarshal(body, &reused); err != nil {
			return err
		}
		if reused.Source != source || reused.CandidateCommit != commit || reused.ScopeRunID != run ||
			!reused.ProductionIteratorVerified || !reused.ConfigRoundTripVerified || !reused.SourceWritesPerformed {
			return errors.New("current candidate's actual serialized Cosmos source preflight required")
		}
	}
	datasets := map[string]any{}
	for _, dataset := range []string{"base", "append", "upsert"} {
		path := map[string]string{"base": paths.Base, "append": paths.Append, "upsert": paths.Upsert}[dataset]
		f, err := fixture.Verify(path)
		if err != nil {
			return err
		}
		m, ok := manifests[dataset]
		if !ok || f.Plan.Phase != fixture.PhaseP0 || m.FixtureRoot != f.RootSHA256 {
			return errors.New("immutable P0 source provenance differs")
		}
		records := f.Plan.VertexTotal + f.Plan.EdgeTotal
		if !seed && (reused.Datasets[dataset].Records != records || reused.Datasets[dataset].FixtureRootSHA256 != f.RootSHA256) {
			return errors.New("reused Cosmos source fixture provenance differs")
		}
		job := tinyJob(m, directories[dataset], source, 17)
		if err := horizonP0CosmosJob(&job, source, run, dataset, records); err != nil {
			return err
		}
		configPath := filepath.Join(output, "cosmos-"+dataset+"-source.json")
		writeTinyJSON(t, configPath, job)
		job, err = config.Load(configPath)
		if err != nil {
			return err
		}
		files := map[string]int64{}
		hashes := map[string]string{}
		for _, table := range m.Tables {
			file, err := os.Open(filepath.Join(directories[dataset], table.Documents))
			if err != nil {
				return err
			}
			hash := sha256.New()
			_, copyErr := io.Copy(hash, file)
			if err := errors.Join(copyErr, file.Close()); err != nil {
				return err
			}
			if hex.EncodeToString(hash.Sum(nil)) != table.DocumentsHash {
				return errors.New("immutable Cosmos document hash differs before source writes")
			}
			files[table.Documents] = table.Rows
			hashes[table.Documents] = table.DocumentsHash
		}
		container, err := horizonP0CosmosContainer(source, run, dataset)
		if err != nil {
			return err
		}
		var seeded map[string]any
		if seed {
			binary := os.Getenv("AGEFREIGHTER_HORIZON_COSMOS_SEED_BINARY")
			if !filepath.IsAbs(binary) || filepath.Base(binary) != "cosmosfixtureload" {
				return errors.New("current native fixture seeder executable required")
			}
			command := exec.CommandContext(ctx, binary, "--endpoint", endpoint, "--database", "agefreighter",
				"--container", container, "--input", directories[dataset], "--workers", "16", "--require-empty",
				"--max-request-charge", "10000000", "--timeout", "30m")
			command.Stderr = os.Stderr
			body, err := command.Output()
			if err != nil {
				return fmt.Errorf("bounded Cosmos fixture preparation: %w", err)
			}
			var proof struct {
				Rows, RemoteRows int64
				Files            map[string]int64
				Hashes           map[string]string `json:"consumedFileSha256"`
				RequestCharge    float64           `json:"observedRequestCharge"`
				ChargeLimit      float64           `json:"requestChargeLimit"`
			}
			if err := json.Unmarshal(body, &proof); err != nil {
				return err
			}
			if proof.Rows != records || proof.RemoteRows != records || !reflect.DeepEqual(files, proof.Files) ||
				!reflect.DeepEqual(hashes, proof.Hashes) || proof.ChargeLimit != 10000000 ||
				proof.RequestCharge <= 0 || proof.RequestCharge > proof.ChargeLimit {
				return errors.New("actual Cosmos consumed hashes or created/table/aggregate counts differ")
			}
			if err := json.Unmarshal(body, &seeded); err != nil {
				return err
			}
		}
		client, err := cosmossource.NewSDKQueryClient(ctx, endpoint, "agefreighter")
		if err != nil {
			return err
		}
		observed := &horizonCosmosQueries{SDKQueryClient: client}
		verificationErr := verifyHorizonP0CosmosIterator(ctx, job, m, observed, container, records)
		if err := errors.Join(verificationErr, client.Close()); err != nil {
			return err
		}
		if _, err := fixture.Verify(path); err != nil {
			return err
		}
		datasets[dataset] = map[string]any{"container": container, "records": records, "fixtureRootSha256": f.RootSHA256,
			"tablesVerified": len(m.Tables), "seed": seeded, "queryRequestCharge": observed.charge,
			"queryThrottledResponses": observed.ThrottledRequests(), "queryPages": observed.pages}
	}
	writeTinyJSON(t, filepath.Join(output, "cosmos-source-seed.json"), map[string]any{
		"source": source, "candidateCommit": commit, "scopeRunID": run, "sourceWritesPerformed": seed,
		"productionIteratorVerified": true, "configRoundTripVerified": true, "datasets": datasets,
	})
	return nil
}

type horizonCosmosQueries struct {
	*cosmossource.SDKQueryClient
	charge float64
	pages  int
}

func (client *horizonCosmosQueries) NewQueryPager(container, query string, params []cosmossource.Parameter, options cosmossource.QueryOptions) (cosmossource.Pager, error) {
	pager, err := client.SDKQueryClient.NewQueryPager(container, query, params, options)
	if err != nil {
		return nil, err
	}
	return &horizonCosmosPager{Pager: pager, client: client}, nil
}

type horizonCosmosPager struct {
	cosmossource.Pager
	client *horizonCosmosQueries
}

func (pager *horizonCosmosPager) NextPage(ctx context.Context) (cosmossource.Page, error) {
	page, err := pager.Pager.NextPage(ctx)
	if err != nil {
		return page, err
	}
	pager.client.pages++
	pager.client.charge += page.RequestCharge
	if math.IsNaN(page.RequestCharge) || math.IsInf(page.RequestCharge, 0) || page.RequestCharge < 0 ||
		pager.client.charge > 2_000_000 {
		return cosmossource.Page{}, errors.New("Cosmos source-query RU budget exceeded or invalid")
	}
	return page, nil
}

func verifyHorizonP0CosmosIterator(ctx context.Context, job config.LoadJob, manifest portable.Manifest,
	client *horizonCosmosQueries, container string, expected int64,
) error {
	for _, table := range manifest.Tables {
		pager, err := client.NewQueryPager(container, "SELECT VALUE 1 FROM c WHERE c.label = @label",
			[]cosmossource.Parameter{{Name: "@label", Value: table.Name}}, cosmossource.QueryOptions{PageSizeHint: 1000})
		if err != nil {
			return err
		}
		var rows int64
		for pager.More() {
			page, err := pager.NextPage(ctx)
			if err != nil {
				return err
			}
			for _, item := range page.Items {
				var marker int
				if err := json.Unmarshal(item, &marker); err != nil || marker != 1 {
					return errors.New("invalid Cosmos table-count projection")
				}
				rows++
			}
		}
		if rows != table.Rows {
			return errors.New("actual Cosmos source table count differs")
		}
	}
	source := *job.Source.Cosmos
	if source.Gremlin != nil {
		resolved, err := cosmossource.InterpretGremlinDocuments(ctx, source, client)
		if err != nil {
			return err
		}
		source = resolved
	}
	it, err := cosmossource.NewIterator(ctx, cosmossource.IteratorOptions{
		Namespace: job.Source.Namespace, Source: source, Client: client, MaxProperties: 64,
	})
	if err != nil {
		return err
	}
	var rows int64
	for {
		_, err := it.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return errors.Join(err, it.Close())
		}
		rows++
	}
	if rows != expected {
		return errors.Join(errors.New("production Cosmos iterator count differs"), it.Close())
	}
	return it.Close()
}

func TestHorizonP0CosmosConfigScope(t *testing.T) {
	endpoint := "https://afht20261118ignrcapglbrq2.documents.azure.com:443/"
	t.Setenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT", endpoint)
	for _, invalid := range []string{"http://afht20261118ignrcapglbrq2.documents.azure.com", endpoint + "?other=account",
		"https://other.documents.azure.com/", endpoint + "other"} {
		if validateHorizonP0CosmosEndpoint(invalid) == nil {
			t.Fatal("accepted an unrelated Cosmos endpoint")
		}
	}
	manifest := portable.Manifest{Tables: []portable.Table{
		{Name: "Supplier", Kind: "node", Columns: []string{"source_key", "external_id", "name"}},
		{Name: "Product", Kind: "node", Columns: []string{"source_key", "external_id", "name"}},
		{Name: "SUPPLIES", Kind: "edge", StartLabel: "Supplier", EndLabel: "Product",
			Columns: []string{"source_key", "relationship_id", "start_id", "end_id", "quantity"}},
	}}
	for _, source := range []string{"cosmos", "gremlin"} {
		for _, dataset := range []string{"base", "append", "upsert"} {
			job := tinyJob(manifest, t.TempDir(), source, 17)
			records := int64(4580)
			if dataset == "base" {
				records = 458000
			}
			if err := horizonP0CosmosJob(&job, source, "r20261119p0c01", dataset, records); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "job.json")
			writeTinyJSON(t, path, job)
			loaded, err := config.Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(job.Source.Cosmos, loaded.Source.Cosmos) {
				t.Fatal("serialized source configuration differs from the preflight")
			}
			container, err := horizonP0CosmosContainer(source, "r20261119p0c01", dataset)
			if err != nil || container == "tiny-explicit" || container == "tiny-gremlin" {
				t.Fatal("source fixture escapes its immutable scope")
			}
			if err := horizonP0CosmosJob(&job, source, "r20261119p0c01", dataset, records+1); err == nil {
				t.Fatal("accepted a fixture outside the approved exact dataset size")
			}
		}
	}
	for _, source := range []string{"csv", "neo4j44", "cosmos/other"} {
		if _, err := horizonP0CosmosContainer(source, "r20261119p0c01", "base"); err == nil {
			t.Fatal("accepted an unrelated fixture kind")
		}
	}
}
