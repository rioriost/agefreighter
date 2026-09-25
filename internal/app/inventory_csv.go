package app

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	sourcecsv "github.com/rioriost/agefreighter/internal/source/csv"
	"github.com/rioriost/agefreighter/pkg/model"
)

// Inventory deliberately does not inherit profile's bounded-prefix row limit.
// These independent hard limits fail closed, never returning partial exact totals.
const csvInventoryMaxRows int64 = 100_000_000
const csvInventoryMaxBytes int64 = 10 << 30

func csvSourceInventory(ctx context.Context, job config.LoadJob, options InventoryOptions) (report.Document, error) {
	ctx, cancel := context.WithTimeout(ctx, min(time.Duration(job.Runtime.OperationTimeout), 30*time.Minute))
	defer cancel()
	before, inputBytes, err := csvInventorySnapshot(ctx, *job.Source.CSV)
	if err != nil {
		return report.Document{}, err
	}
	iterator, err := sourcecsv.NewIterator(ctx, sourcecsv.IteratorOptions{
		Namespace: job.Source.Namespace, Source: *job.Source.CSV, OptimizeRFC4180: true,
		// RejectLimit stays zero even when a LoadJob permits quarantine.
	})
	if err != nil {
		return report.Document{}, errors.New("CSV inventory initialization failed")
	}
	defer iterator.Close()
	acc := &profileAccumulator{}
	counts := map[string]int64{}
	for _, v := range job.Source.CSV.Vertices {
		counts["vertex:"+v.Label] = 0
	}
	for _, e := range job.Source.CSV.Edges {
		counts["edge:"+e.Label] = 0
	}
	for {
		item, readErr := iterator.Next(ctx)
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return report.Document{}, errors.New("CSV inventory did not complete; malformed, changed, unreadable or canceled source")
		}
		if acc.rows >= csvInventoryMaxRows {
			return report.Document{}, errors.New("CSV inventory row limit exceeded; no exact result")
		}
		acc.rows++
		acc.bytes = saturatingProfileAdd(acc.bytes, profileRecordWidth(item.Record))
		switch item.Record.Kind() {
		case model.RecordVertex:
			acc.vertices++
			counts["vertex:"+string(item.Record.Vertex.Label)]++
		case model.RecordEdge:
			acc.edges++
			counts["edge:"+string(item.Record.Edge.Label)]++
		default:
			return report.Document{}, errors.New("CSV inventory received an invalid record")
		}
	}
	if err := iterator.Close(); err != nil {
		return report.Document{}, errors.New("CSV inventory source close failed")
	}
	after, _, err := csvInventorySnapshot(ctx, *job.Source.CSV)
	if err != nil || before != after {
		return report.Document{}, errors.New("CSV inventory source changed or final fingerprint could not be verified")
	}
	at := options.GeneratedAt
	if at.IsZero() {
		at = time.Now()
	}
	doc := report.New("inventory", at)
	doc.Outcome = report.OutcomePass
	doc.Checks = []report.Check{
		{ID: "source-counts", Status: report.CheckPass, Summary: "all configured CSV mappings reached EOF without rejected records"},
		{ID: "source-unchanged", Status: report.CheckPass, Summary: "all mapped file bytes and metadata matched before and after the complete scan"},
		{ID: "read-only", Status: report.CheckPass, Summary: "source-only scan; target credentials and quarantine were not used"},
	}
	doc.Warnings = append(doc.Warnings, report.Finding{Code: "INVENTORY_NOT_MIGRATION_VERIFICATION", Message: "Counts describe mapped records, not unique identities or endpoint existence. Capacity ranges are estimates, not a migration or sizing approval."})
	fields := []report.Field{
		passField("connector", "csv"), passField("countMethod", "csv-complete-stream"),
		passField("vertices", strconv.FormatInt(acc.vertices, 10)), passField("edges", strconv.FormatInt(acc.edges, 10)),
		passField("totalRows", strconv.FormatInt(acc.rows, 10)), passField("inputBytes", strconv.FormatInt(inputBytes, 10)),
		passField("mappedRecordBytes", strconv.FormatInt(acc.bytes, 10)), passField("sourceFingerprint", before),
	}
	labels := report.Section{Title: "Mapped record counts"}
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		labels.Fields = append(labels.Fields, passField(key, strconv.FormatInt(counts[key], 10)))
	}
	capacity := profileCapacitySection(profileRun{job: job, accumulator: acc, complete: true})
	// Inventory has no throughput evidence. Omit migration-time predictions
	// rather than treating an unavailable prediction as an exact-count failure.
	capacity.Fields = slices.DeleteFunc(capacity.Fields, func(f report.Field) bool { return f.Name == "estimatedMigrationTime" })
	doc.Sections = []report.Section{{Title: "Source inventory", Fields: fields}, labels, capacity}
	if _, err := report.Render(doc, report.FormatJSON); err != nil {
		return report.Document{}, err
	}
	return doc, nil
}

// No paths or row values are emitted in inventory reports. Deduplicate physical
// paths for input byte totals, but count records once per configured mapping.
func csvInventorySnapshot(ctx context.Context, source config.CSVSource) (string, int64, error) {
	paths := map[string]bool{}
	for _, v := range source.Vertices {
		paths[v.Path] = true
	}
	for _, e := range source.Edges {
		paths[e.Path] = true
	}
	if len(paths) == 0 || len(paths) > 64 {
		return "", 0, errors.New("CSV inventory requires 1..64 files")
	}
	ordered := make([]string, 0, len(paths))
	for path := range paths {
		ordered = append(ordered, path)
	}
	slices.Sort(ordered)
	digest := sha256.New()
	total := int64(0)
	buffer := make([]byte, 64<<10)
	for _, path := range ordered {
		if err := ctx.Err(); err != nil {
			return "", 0, err
		}
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() || info.Size() > csvInventoryMaxBytes-total {
			return "", 0, errors.New("CSV inventory requires regular files totaling at most 10 GiB")
		}
		f, err := os.Open(path)
		if err != nil {
			return "", 0, errors.New("CSV inventory file unavailable")
		}
		opened, err := f.Stat()
		if err != nil || !os.SameFile(info, opened) {
			f.Close()
			return "", 0, errors.New("CSV inventory file identity changed")
		}
		_, _ = fmt.Fprintf(digest, "%d:%s:%d:%d:", len(path), path, info.Size(), info.ModTime().UnixNano())
		n := int64(0)
		for {
			if err = ctx.Err(); err != nil {
				break
			}
			var count int
			count, err = f.Read(buffer)
			n += int64(count)
			if n > info.Size() {
				err = errors.New("file grew")
				break
			}
			_, _ = digest.Write(buffer[:count])
			if err != nil {
				break
			}
		}
		end, statErr := f.Stat()
		closeErr := f.Close()
		if !errors.Is(err, io.EOF) || statErr != nil || closeErr != nil || n != info.Size() || end.Size() != info.Size() || !end.ModTime().Equal(info.ModTime()) {
			return "", 0, errors.New("CSV inventory fingerprint failed or file changed")
		}
		total += n
	}
	return hex.EncodeToString(digest.Sum(nil)), total, nil
}
