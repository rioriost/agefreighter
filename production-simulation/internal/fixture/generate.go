package fixture

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type GenerateConfig struct {
	Phase   Phase
	Output  string
	Shards  int
	Workers int
	Seed    uint64
}

type FileEntry struct {
	Path   string `json:"path"`
	Kind   string `json:"kind"`
	Name   string `json:"name"`
	Rows   int64  `json:"rows"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type Manifest struct {
	Version    int         `json:"version"`
	CreatedAt  string      `json:"createdAt"`
	Seed       uint64      `json:"seed"`
	Shards     int         `json:"shards"`
	Plan       Plan        `json:"plan"`
	Files      []FileEntry `json:"files"`
	RootSHA256 string      `json:"rootSha256"`
}

type generationTask struct {
	kind  string
	name  string
	shard int
}

func Generate(ctx context.Context, config GenerateConfig) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, errors.New("context is required")
	}
	if config.Output == "" {
		return Manifest{}, errors.New("output directory is required")
	}
	if config.Shards < 1 || config.Shards > 4096 {
		return Manifest{}, fmt.Errorf("shards must be from 1 to 4096")
	}
	if config.Workers == 0 {
		config.Workers = runtime.NumCPU()
	}
	if config.Workers < 1 || config.Workers > 256 {
		return Manifest{}, fmt.Errorf("workers must be from 1 to 256")
	}
	plan, err := BuildPlan(config.Phase)
	if err != nil {
		return Manifest{}, err
	}
	if err := os.MkdirAll(filepath.Dir(config.Output), 0o750); err != nil {
		return Manifest{}, fmt.Errorf("create output parent: %w", err)
	}
	if err := os.Mkdir(config.Output, 0o750); err != nil {
		if errors.Is(err, os.ErrExist) {
			return Manifest{}, fmt.Errorf("output directory already exists: %s", config.Output)
		}
		return Manifest{}, fmt.Errorf("create output directory: %w", err)
	}
	for _, directory := range []string{"headers/nodes", "headers/edges", "nodes", "edges"} {
		if err := os.MkdirAll(filepath.Join(config.Output, directory), 0o750); err != nil {
			return Manifest{}, fmt.Errorf("create fixture directory: %w", err)
		}
	}

	entries, err := writeHeaders(config.Output, plan)
	if err != nil {
		return Manifest{}, err
	}

	workerContext, cancel := context.WithCancel(ctx)
	defer cancel()
	tasks := make(chan generationTask)
	results := make(chan FileEntry)
	errorsChannel := make(chan error, 1)
	var workers sync.WaitGroup
	for worker := 0; worker < config.Workers; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for task := range tasks {
				entry, taskErr := writeTask(workerContext, config, plan, task)
				if taskErr != nil {
					select {
					case errorsChannel <- taskErr:
					default:
					}
					cancel()
					return
				}
				if entry.Rows > 0 {
					select {
					case results <- entry:
					case <-workerContext.Done():
						return
					}
				}
			}
		}()
	}

	go func() {
		defer close(tasks)
		for _, item := range plan.VertexSpecs {
			for shard := 0; shard < config.Shards; shard++ {
				select {
				case tasks <- generationTask{kind: "node", name: item.Label, shard: shard}:
				case <-workerContext.Done():
					return
				}
			}
		}
		for _, item := range plan.EdgeSpecs {
			for shard := 0; shard < config.Shards; shard++ {
				select {
				case tasks <- generationTask{kind: "edge", name: item.Type, shard: shard}:
				case <-workerContext.Done():
					return
				}
			}
		}
	}()

	done := make(chan struct{})
	go func() {
		workers.Wait()
		close(results)
		close(done)
	}()
	for entry := range results {
		entries = append(entries, entry)
	}
	<-done
	select {
	case generationErr := <-errorsChannel:
		return Manifest{}, generationErr
	default:
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}

	sort.Slice(entries, func(left, right int) bool { return entries[left].Path < entries[right].Path })
	manifest := Manifest{
		Version: ManifestVersion, CreatedAt: time.Now().UTC().Format(time.RFC3339),
		Seed: config.Seed, Shards: config.Shards, Plan: plan, Files: entries,
	}
	manifest.RootSHA256 = manifestRoot(entries)
	if err := writeManifest(filepath.Join(config.Output, "manifest.json"), manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func writeHeaders(output string, plan Plan) ([]FileEntry, error) {
	entries := make([]FileEntry, 0, len(plan.VertexSpecs)+len(plan.EdgeSpecs))
	for _, item := range plan.VertexSpecs {
		path := filepath.Join("headers", "nodes", item.Label+".header.csv")
		entry, err := writeRows(output, path, "node-header", item.Label, [][]string{{
			"source_key:ID", "external_id:string", "name:string", "region:string",
			"created_at:datetime", "status:string", "score:double", "active:boolean",
			"tags:string[]", "quantities:long[]", "description:string",
		}})
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	for _, item := range plan.EdgeSpecs {
		path := filepath.Join("headers", "edges", item.Type+".header.csv")
		entry, err := writeRows(output, path, "edge-header", item.Type, [][]string{{
			"source_key:long", "relationship_id:string", ":START_ID", ":END_ID",
			"occurred_at:datetime", "quantity:long", "status:string", "distance_km:double",
			"notes:string",
		}})
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func writeTask(ctx context.Context, config GenerateConfig, plan Plan, task generationTask) (FileEntry, error) {
	if task.kind == "node" {
		for _, item := range plan.VertexSpecs {
			if item.Label == task.name {
				return writeNodeShard(ctx, config, item, task.shard)
			}
		}
	}
	if task.kind == "edge" {
		for _, item := range plan.EdgeSpecs {
			if item.Type == task.name {
				return writeEdgeShard(ctx, config, plan, item, task.shard)
			}
		}
	}
	return FileEntry{}, fmt.Errorf("unknown generation task %s/%s", task.kind, task.name)
}

func writeNodeShard(ctx context.Context, config GenerateConfig, item VertexSpec, shard int) (FileEntry, error) {
	start, end := partition(item.Count, config.Shards, shard)
	if start == end {
		return FileEntry{}, nil
	}
	path := filepath.Join("nodes", item.Label, fmt.Sprintf("part-%05d.csv", shard))
	return writeGenerated(outputPath(config.Output, path), path, "node", item.Label, end-start,
		func(writer *csv.Writer, digest hash.Hash) error {
			_ = digest
			for local := start; local < end; local++ {
				if local&0x3fff == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				key := item.FirstKey + local
				hashed := splitMix64(config.Seed ^ uint64(key)*0x9e3779b97f4a7c15)
				status := []string{"active", "pending", "suspended", "archived"}[hashed%4]
				if hashed%97 == 0 {
					status = ""
				}
				row := []string{
					strconv.FormatInt(key, 10), externalID(item.Label, local),
					fmt.Sprintf("%s-%d-東京", item.Label, local+1), regions[(hashed>>8)%uint64(len(regions))],
					timestamp(hashed), status,
					strconv.FormatFloat(float64(hashed%1_000_000)/10_000, 'f', 4, 64),
					strconv.FormatBool(hashed%11 != 0),
					fmt.Sprintf("tier-%d;segment-%d", hashed%5, (hashed>>12)%17),
					fmt.Sprintf("%d;%d;%d", 1+hashed%7, 1+(hashed>>7)%31, 1+(hashed>>14)%127),
					description(item.Label, hashed),
				}
				if err := writer.Write(row); err != nil {
					return err
				}
			}
			return nil
		})
}

func writeEdgeShard(ctx context.Context, config GenerateConfig, plan Plan, item EdgeSpec, shard int) (FileEntry, error) {
	start, end := partition(item.Count, config.Shards, shard)
	if start == end {
		return FileEntry{}, nil
	}
	vertices := make(map[string]VertexSpec, len(plan.VertexSpecs))
	for _, vertex := range plan.VertexSpecs {
		vertices[vertex.Label] = vertex
	}
	startVertex, startOK := vertices[item.Start]
	endVertex, endOK := vertices[item.End]
	if !startOK || !endOK || startVertex.Count == 0 || endVertex.Count == 0 {
		return FileEntry{}, fmt.Errorf("edge %s has an empty endpoint label", item.Type)
	}
	path := filepath.Join("edges", item.Type, fmt.Sprintf("part-%05d.csv", shard))
	return writeGenerated(outputPath(config.Output, path), path, "edge", item.Type, end-start,
		func(writer *csv.Writer, digest hash.Hash) error {
			_ = digest
			for local := start; local < end; local++ {
				if local&0x3fff == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				key := item.FirstKey + local
				hashed := splitMix64(config.Seed ^ uint64(key)*0xd6e8feb86659fd93 ^ hashString(item.Type))
				startLocal := endpointIndex(hashed, startVertex.Count, isSkewed(item.Start))
				endLocal := endpointIndex(splitMix64(hashed), endVertex.Count, isSkewed(item.End))
				row := []string{
					strconv.FormatInt(key, 10), relationshipID(item.Type, local),
					strconv.FormatInt(startVertex.FirstKey+startLocal, 10),
					strconv.FormatInt(endVertex.FirstKey+endLocal, 10),
					timestamp(hashed), strconv.FormatUint(1+(hashed%10_000), 10),
					[]string{"planned", "in_transit", "complete", "exception"}[(hashed>>9)%4],
					strconv.FormatFloat(float64(hashed%2_000_000)/100, 'f', 2, 64),
					description(item.Type, hashed>>3),
				}
				if err := writer.Write(row); err != nil {
					return err
				}
			}
			return nil
		})
}

func writeRows(output, relative, kind, name string, rows [][]string) (FileEntry, error) {
	return writeGenerated(outputPath(output, relative), relative, kind, name, int64(len(rows)),
		func(writer *csv.Writer, digest hash.Hash) error {
			_ = digest
			writer.WriteAll(rows)
			return writer.Error()
		})
}

func writeGenerated(absolute, relative, kind, name string, rows int64, emit func(*csv.Writer, hash.Hash) error) (FileEntry, error) {
	if err := os.MkdirAll(filepath.Dir(absolute), 0o750); err != nil {
		return FileEntry{}, fmt.Errorf("create output parent: %w", err)
	}
	file, err := os.OpenFile(absolute, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o640)
	if err != nil {
		return FileEntry{}, fmt.Errorf("create %s: %w", relative, err)
	}
	digest := sha256.New()
	buffered := bufio.NewWriterSize(io.MultiWriter(file, digest), 256*1024)
	writer := csv.NewWriter(buffered)
	emitErr := emit(writer, digest)
	writer.Flush()
	if emitErr == nil {
		emitErr = writer.Error()
	}
	if emitErr == nil {
		emitErr = buffered.Flush()
	}
	closeErr := file.Close()
	if emitErr != nil {
		return FileEntry{}, fmt.Errorf("write %s: %w", relative, emitErr)
	}
	if closeErr != nil {
		return FileEntry{}, fmt.Errorf("close %s: %w", relative, closeErr)
	}
	info, err := os.Stat(absolute)
	if err != nil {
		return FileEntry{}, fmt.Errorf("stat %s: %w", relative, err)
	}
	return FileEntry{
		Path: filepath.ToSlash(relative), Kind: kind, Name: name, Rows: rows,
		Bytes: info.Size(), SHA256: hex.EncodeToString(digest.Sum(nil)),
	}, nil
}

func writeManifest(path string, manifest Manifest) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o640)
	if err != nil {
		return fmt.Errorf("create manifest: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	err = encoder.Encode(manifest)
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

func manifestRoot(entries []FileEntry) string {
	digest := sha256.New()
	for _, entry := range entries {
		fmt.Fprintf(digest, "%s\x00%s\x00%d\x00%d\n", entry.Path, entry.SHA256, entry.Rows, entry.Bytes)
	}
	return hex.EncodeToString(digest.Sum(nil))
}

func partition(total int64, shards, shard int) (int64, int64) {
	return total * int64(shard) / int64(shards), total * int64(shard+1) / int64(shards)
}

func endpointIndex(value uint64, count int64, skew bool) int64 {
	if !skew {
		return int64(value % uint64(count))
	}
	other := splitMix64(value)
	return int64(min(value%uint64(count), other%uint64(count)))
}

func isSkewed(label string) bool {
	return label == "Supplier" || label == "Facility" || label == "Product" || label == "Carrier"
}

func splitMix64(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func hashString(value string) uint64 {
	var result uint64 = 1469598103934665603
	for _, character := range []byte(value) {
		result ^= uint64(character)
		result *= 1099511628211
	}
	return result
}

func externalID(label string, local int64) string {
	return strings.ToLower(label) + "-" + fmt.Sprintf("%012d", local+1)
}

func relationshipID(typeName string, local int64) string {
	return strings.ToLower(typeName) + "-" + fmt.Sprintf("%015d", local+1)
}

func timestamp(value uint64) string {
	const secondsInFiveYears = uint64(5 * 365 * 24 * 60 * 60)
	return time.Unix(1_577_836_800+int64(value%secondsInFiveYears), 0).UTC().Format(time.RFC3339)
}

func description(prefix string, value uint64) string {
	length := 32
	switch bucket := value % 1000; {
	case bucket >= 999:
		length = 8192
	case bucket >= 990:
		length = 2048
	case bucket >= 900:
		length = 256
	}
	base := prefix + "-representative-"
	if len(base) >= length {
		return base[:length]
	}
	return base + strings.Repeat("x", length-len(base))
}

func outputPath(root, relative string) string {
	return filepath.Join(root, filepath.FromSlash(relative))
}

var regions = []string{"JP-13", "JP-27", "US-WA", "DE-BE", "SG-01", "日本-東"}
