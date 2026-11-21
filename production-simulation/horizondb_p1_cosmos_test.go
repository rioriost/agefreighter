//go:build linux || darwin

package simulation_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
	"github.com/rioriost/agefreighter/internal/config"
	cosmossource "github.com/rioriost/agefreighter/internal/source/cosmos"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

const (
	p1CosmosDatabase      = "agefreighter"
	p1CosmosSeedWriteRate = 1800
)

func TestHorizonDBP1CosmosPG17Integration(t *testing.T)  { runHorizonDBP1(t, 17, "cosmos") }
func TestHorizonDBP1CosmosPG18Integration(t *testing.T)  { runHorizonDBP1(t, 18, "cosmos") }
func TestHorizonDBP1GremlinPG17Integration(t *testing.T) { runHorizonDBP1(t, 17, "gremlin") }
func TestHorizonDBP1GremlinPG18Integration(t *testing.T) { runHorizonDBP1(t, 18, "gremlin") }

func p1IsCosmos(source string) bool { return source == "cosmos" || source == "gremlin" }

func p1CosmosEndpoint(run string) string {
	return "https://afhsp1" + run + ".documents.azure.com:443/"
}

func p1CosmosResource(subscription, run string) string {
	return "/subscriptions/" + subscription + "/resourceGroups/rg-agefreighter-horizon-scale-20261118/providers/Microsoft.DocumentDB/databaseAccounts/afhsp1" + run
}

func p1CosmosJob(m portable.Manifest, a p1Admission, dataset string) (config.LoadJob, error) {
	if !p1IsCosmos(a.source()) || !p1RunPattern.MatchString(a.RunID) || !slices.Contains(p1FixtureOrder[:], dataset) {
		return config.LoadJob{}, errors.New("require the admitted Cosmos account and dataset")
	}
	job := tinyJob(m, "", a.source(), a.writerMajor())
	source := job.Source.Cosmos
	source.Endpoint, source.Database, source.PageSize = p1CosmosEndpoint(a.RunID), p1CosmosDatabase, 1000
	if source.Gremlin != nil {
		// Indexed exclusion discovers catalog entries, not one row per document.
		source.Gremlin.Container, source.Gremlin.MaxDiscoveryDocuments = dataset, 100_000
	}
	for i := range source.Vertices {
		source.Vertices[i].Container = dataset
	}
	for i := range source.Edges {
		source.Edges[i].Container = dataset
	}
	return job, nil
}

func validateP1CosmosWorker(t *testing.T, job config.LoadJob, a p1Admission, dataset, fixtureRoot string) error {
	t.Helper()
	var manifest portable.Manifest
	p1ReadJSON(t, filepath.Join(a.Output, "csv-"+dataset, portable.ManifestFilename), &manifest)
	if manifest.FixtureRoot == "" || manifest.FixtureRoot != fixtureRoot {
		return errors.New("Cosmos loader fixture root differs from its admitted dataset")
	}
	want, err := p1CosmosJob(manifest, a, dataset)
	if err != nil {
		return err
	}
	if job.Source.Type != config.SourceCosmos || !reflect.DeepEqual(job.Source.Cosmos, want.Source.Cosmos) {
		return errors.New("Cosmos loader differs from its leased account, dataset or document mappings")
	}
	return nil
}

func requireP1CosmosPrivate(t *testing.T, ctx context.Context, a p1Admission) {
	t.Helper()
	if os.Getenv("AZURE_TOKEN_CREDENTIALS") != "ManagedIdentityCredential" || os.Getenv("AZURE_CLIENT_ID") != "" {
		t.Fatal("require the Cosmos worker system identity without the SQL user-assigned identity")
	}
	addresses, err := net.DefaultResolver.LookupIPAddr(ctx, "afhsp1"+a.RunID+".documents.azure.com")
	_, vnet, parseErr := net.ParseCIDR("10.86.0.0/16")
	if err != nil || parseErr != nil || len(addresses) == 0 {
		t.Fatal("leased Cosmos private DNS is unavailable")
	}
	for _, address := range addresses {
		if !vnet.Contains(address.IP) {
			t.Fatal("leased Cosmos endpoint resolves outside the private VNet")
		}
	}
}

type p1CosmosSeed struct {
	Rows, RemoteRows int64
	Files            map[string]int64
	Hashes           map[string]string `json:"consumedFileSha256"`
	RequestCharge    float64           `json:"observedRequestCharge"`
	Throttled        int64             `json:"throttledResponses"`
	ChargeLimit      float64           `json:"requestChargeLimit"`
	MaxWrites        int               `json:"maxWritesPerSecond"`
}

type p1CosmosSeedReceipt struct {
	RunID    string                  `json:"runId"`
	Source   string                  `json:"source"`
	Commit   string                  `json:"candidateCommit"`
	Datasets map[string]p1CosmosSeed `json:"datasets"`
}

func TestHorizonDBP1SeedCosmos(t *testing.T) {
	root := os.Getenv("AGEFREIGHTER_HORIZON_P1_COSMOS_SEED_DIR")
	if root == "" {
		t.Skip("explicit P1 Cosmos preparation required")
	}
	a := p1Admission{
		RunID:      os.Getenv("AGEFREIGHTER_HORIZON_P1_COSMOS_RUN"),
		Source:     os.Getenv("AGEFREIGHTER_HORIZON_P1_COSMOS_SOURCE"),
		Commit:     os.Getenv("AGEFREIGHTER_HORIZON_P1_PREPARE_COMMIT"),
		FixtureDir: filepath.Join(root, "fixture"),
	}
	subscription := os.Getenv("AGEFREIGHTER_HORIZON_P1_COSMOS_SUBSCRIPTION")
	if major, _ := p1EnabledRoute(t); major != 0 || os.Getenv(p1ReceiptEnv) != "" ||
		!p1IsCosmos(a.source()) || !p1RunPattern.MatchString(a.RunID) ||
		!p1CommitPattern.MatchString(a.Commit) || !p1UUIDPattern.MatchString(subscription) ||
		root != "/var/lib/agefreighter-p1/"+a.RunID {
		t.Fatal("require an exact source-only P1 Cosmos worker and candidate")
	}
	ctx, stop := context.WithTimeout(t.Context(), 80*time.Minute)
	defer stop()
	requireP1CosmosPrivate(t, ctx, a)
	monitor := startP1CosmosTelemetry(t, ctx, stop, p1CosmosResource(subscription, a.RunID), root)
	defer func() { monitor(true) }()
	paths, _, specs := verifiedP1Fixtures(t, ctx, a.FixtureDir)
	reports := make(map[string]p1CosmosSeed, len(p1FixtureOrder))
	for _, dataset := range p1FixtureOrder {
		dir := filepath.Join(root, "cosmos-"+dataset)
		export := portable.Export
		if a.source() == "gremlin" {
			export = portable.ExportGremlin
		}
		manifest, err := export(ctx, paths[dataset], dir)
		if err != nil {
			t.Fatal(err)
		}
		limit := "1100000"
		if dataset == "base" {
			limit = "110000000"
		}
		command := exec.CommandContext(ctx, filepath.Join(root, "cosmosfixtureload"),
			"--endpoint", p1CosmosEndpoint(a.RunID), "--database", p1CosmosDatabase, "--container", dataset,
			"--input", dir, "--workers", "64", "--require-empty", "--max-request-charge", limit,
			"--max-writes-per-second", strconv.Itoa(p1CosmosSeedWriteRate), "--timeout", "70m")
		command.Stderr = os.Stderr
		body, err := command.Output()
		if err != nil {
			t.Fatalf("P1 Cosmos %s seed failed: %v", dataset, err)
		}
		var proof p1CosmosSeed
		chargeLimit := float64(1_100_000)
		if dataset == "base" {
			chargeLimit *= 100
		}
		spec := specs[dataset]
		if err := json.Unmarshal(body, &proof); err != nil || proof.Rows != spec.Vertices+spec.Edges ||
			proof.RemoteRows != proof.Rows || proof.ChargeLimit != chargeLimit || proof.RequestCharge <= 0 ||
			proof.RequestCharge > chargeLimit || proof.Throttled < 0 || proof.MaxWrites != p1CosmosSeedWriteRate ||
			len(proof.Files) != len(manifest.Tables) || len(proof.Hashes) != len(manifest.Tables) {
			t.Fatal("Cosmos seed did not prove admitted counts or RU usage")
		}
		for _, table := range manifest.Tables {
			if proof.Files[table.Documents] != table.Rows || proof.Hashes[table.Documents] != table.DocumentsHash {
				t.Fatal("Cosmos seed did not consume the exact admitted documents")
			}
		}
		reports[dataset] = proof
	}
	writeTinyJSON(t, filepath.Join(root, "cosmos-source-seed.json"), p1CosmosSeedReceipt{
		RunID: a.RunID, Source: a.source(), Commit: a.Commit, Datasets: reports,
	})
	t.Log("Pinned Cosmos datasets seeded and counted")
}

func p1GremlinExpected(t *testing.T, ctx context.Context, paths map[string]string) map[string]rangedigest.Manifest {
	t.Helper()
	roots := map[string]string{
		"base":   "8a048faa36fad90404c263d3ce75073d117e5d96a15f8a614a42347cbd7a0ef4",
		"append": "c81934d06d8db19523976457e17a096eef55849dec7d579509a11c36caa1459c",
		"upsert": "8b0766f527fd6207f99c2771b92dd7a77a42d622e875b659a01b576d62f5ea15",
	}
	manifests := make(map[string]rangedigest.Manifest, len(roots))
	for _, dataset := range p1FixtureOrder {
		digest, err := rangedigest.ScenarioManifest(ctx, p1Scenario(paths, dataset), 100_000, true)
		if err != nil {
			t.Fatal(err)
		}
		if digest.RootSHA256 != roots[dataset] {
			t.Fatal("partition-aware canonical root differs from the frozen P1 oracle")
		}
		manifests[dataset] = digest
	}
	return manifests
}

func requireP1CosmosReadOnly(t *testing.T, ctx context.Context, a p1Admission) {
	t.Helper()
	identity, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		t.Fatal("could not initialize Cosmos worker system identity")
	}
	client, err := azcosmos.NewClient(p1CosmosEndpoint(a.RunID), identity, nil)
	if err != nil {
		t.Fatal("could not initialize Cosmos read-only probe")
	}
	defer client.Close()
	container, err := client.NewContainer(p1CosmosDatabase, "base")
	if err != nil {
		t.Fatal("could not bind read-only probe to the leased base")
	}
	const probe = "p1-read-only-probe"
	_, err = container.ReplaceItem(ctx, azcosmos.NewPartitionKeyString(probe), probe,
		[]byte(`{"id":"p1-read-only-probe","partitionKey":"p1-read-only-probe"}`), nil)
	var denied *azcore.ResponseError
	if !errors.As(err, &denied) || denied.StatusCode != 403 || !strings.Contains(denied.Error(), "items/replace") {
		t.Fatal("Cosmos system identity did not prove a data-plane write denial")
	}
}

type p1CosmosSnapshot struct {
	Tables    []p1SourceTable `json:"tables"`
	Charge    float64         `json:"requestCharge"`
	Pages     int             `json:"pages"`
	Throttled int64           `json:"throttledResponses"`
	ReadOnly  bool            `json:"readOnly"`
}

func snapshotP1Cosmos(ctx context.Context, a p1Admission, exports map[string]portable.Manifest) (p1CosmosSnapshot, error) {
	var snapshot p1CosmosSnapshot
	sdk, err := cosmossource.NewSDKQueryClient(ctx, p1CosmosEndpoint(a.RunID), p1CosmosDatabase)
	if err != nil {
		return snapshot, err
	}
	defer sdk.Close()
	client := &horizonCosmosQueries{SDKQueryClient: sdk, maxCharge: 25_000_000}
	for _, dataset := range p1FixtureOrder {
		for _, table := range exports[dataset].Tables {
			pager, err := client.NewQueryPager(dataset, "SELECT c.id,c.partitionKey,c._etag FROM c WHERE c.label = @label",
				[]cosmossource.Parameter{{Name: "@label", Value: table.Name}}, cosmossource.QueryOptions{PageSizeHint: 1000})
			if err != nil {
				return snapshot, err
			}
			hashes := make([][sha256.Size]byte, 0, table.Rows)
			for pager.More() {
				page, err := pager.NextPage(ctx)
				if err != nil {
					return snapshot, err
				}
				for _, item := range page.Items {
					var doc struct {
						ID  string `json:"id"`
						Key string `json:"partitionKey"`
						Tag string `json:"_etag"`
					}
					if err := json.Unmarshal(item, &doc); err != nil || doc.ID == "" || doc.Key == "" || doc.Tag == "" ||
						int64(len(hashes)) >= table.Rows {
						return snapshot, errors.New("Cosmos snapshot identity, version or count differs")
					}
					encoded, err := json.Marshal(doc)
					if err != nil {
						return snapshot, err
					}
					hashes = append(hashes, sha256.Sum256(encoded))
				}
			}
			if int64(len(hashes)) != table.Rows {
				return snapshot, errors.New("Cosmos snapshot count differs")
			}
			slices.SortFunc(hashes, func(a, b [sha256.Size]byte) int { return bytes.Compare(a[:], b[:]) })
			sum := sha256.New()
			for i, hash := range hashes {
				if i > 0 && hash == hashes[i-1] {
					return snapshot, errors.New("Cosmos snapshot contains a duplicate item")
				}
				sum.Write(hash[:])
			}
			snapshot.Tables = append(snapshot.Tables, p1SourceTable{dataset, table.Name, table.Rows, hex.EncodeToString(sum.Sum(nil))})
		}
	}
	snapshot.Charge, snapshot.Pages, snapshot.Throttled = client.charge, client.pages, client.ThrottledRequests()
	if snapshot.Charge <= 0 || snapshot.Pages < 1 {
		return snapshot, errors.New("Cosmos snapshot requires observed RU and pages")
	}
	return snapshot, nil
}

func prepareP1Cosmos(t *testing.T, ctx context.Context, a p1Admission, exports map[string]portable.Manifest) func() {
	t.Helper()
	var seed p1CosmosSeedReceipt
	seedPath := filepath.Join(filepath.Dir(a.FixtureDir), "cosmos-source-seed.json")
	if p1FileSHA(t, seedPath) != a.CosmosSeedSHA256 {
		t.Fatal("Cosmos seed receipt SHA-256 differs from admission")
	}
	p1ReadJSON(t, seedPath, &seed)
	if seed.RunID != a.RunID || seed.Source != a.source() || seed.Commit != a.Commit || len(seed.Datasets) != len(p1FixtureOrder) {
		t.Fatal("Cosmos seed evidence differs from the admitted route and candidate")
	}
	for _, dataset := range p1FixtureOrder {
		proof, spec := seed.Datasets[dataset], p1FixtureSpecification(dataset)
		if proof.Rows != spec.Vertices+spec.Edges || proof.RemoteRows != proof.Rows ||
			proof.RequestCharge <= 0 || proof.RequestCharge > proof.ChargeLimit || proof.MaxWrites != p1CosmosSeedWriteRate {
			t.Fatal("Cosmos seed counts or charge differ from the admitted dataset")
		}
	}
	requireP1CosmosPrivate(t, ctx, a)
	requireP1CosmosReadOnly(t, ctx, a)
	scan := func(name string) p1CosmosSnapshot {
		scanCtx, stop := context.WithTimeout(ctx, 25*time.Minute)
		defer stop()
		snapshot, err := snapshotP1Cosmos(scanCtx, a, exports)
		if err != nil {
			t.Fatalf("Cosmos %s source snapshot failed (%T)", name, err)
		}
		snapshot.ReadOnly = true
		writeTinyJSON(t, filepath.Join(a.Output, "cosmos-source-"+name+".json"), snapshot)
		return snapshot
	}
	before := scan("before")
	return func() {
		after := scan("after")
		if !reflect.DeepEqual(before.Tables, after.Tables) {
			t.Fatal("Cosmos source counts or document identity/version digests changed")
		}
	}
}

func TestHorizonDBP1CosmosWorkerScope(t *testing.T) {
	a := p1Admission{RunID: "rp1test01", Output: t.TempDir(), Host: "afhs-20261118-pg17.rw.scope.japaneast.horizondb.azure.com"}
	dir := filepath.Join(a.Output, "fixture")
	if _, err := fixture.Generate(t.Context(), fixture.GenerateConfig{
		Phase: fixture.PhaseTiny, Output: dir, Seed: 20260829, Shards: 1, Workers: 1,
	}); err != nil {
		t.Fatal(err)
	}
	m, err := portable.Export(t.Context(), filepath.Join(dir, "manifest.json"), filepath.Join(a.Output, "csv-append"))
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{"cosmos", "gremlin"} {
		t.Run(source, func(t *testing.T) {
			a := a
			a.Source = source
			job, err := p1CosmosJob(m, a, "append")
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(a.Output, source+".json")
			writeTinyJSON(t, path, job)
			loaded, err := config.Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := validateP1CosmosWorker(t, loaded, a, "append", m.FixtureRoot); err != nil {
				t.Fatal(err)
			}
			if validateP1CosmosWorker(t, loaded, a, "append", "unadmitted-root") == nil {
				t.Fatal("accepted a different Cosmos fixture root")
			}
			if loaded.Source.Cosmos.Gremlin != nil {
				loaded.Source.Cosmos.Gremlin.Container = "base"
			} else {
				loaded.Source.Cosmos.Edges[0].Container = "base"
			}
			if validateP1CosmosWorker(t, loaded, a, "append", m.FixtureRoot) == nil {
				t.Fatal("accepted a different Cosmos dataset")
			}
			loaded.Source.Cosmos = job.Source.Cosmos
			loaded.Source.Cosmos.Endpoint = p1CosmosEndpoint("rincorrect")
			if validateP1CosmosWorker(t, loaded, a, "append", m.FixtureRoot) == nil {
				t.Fatal("accepted a different Cosmos account")
			}
		})
	}
}
