package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/rioriost/agefreighter/internal/source/postgres"
	"github.com/spf13/cobra"
)

// Private runner command. It accepts a connection-only review rather than a
// synthetic LoadJob. Credentials remain in the protected worker environment.
func newPostgresCatalogCommand() *cobra.Command {
	return &cobra.Command{
		Use: "postgres-catalog CONFIG", Hidden: true, Args: cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			file, err := os.Open(args[0])
			if err != nil {
				return errors.New("catalog configuration is unavailable")
			}
			defer file.Close()
			info, err := file.Stat()
			if err != nil || !info.Mode().IsRegular() || info.Size() > postgres.CatalogRequestMaxBytes {
				return errors.New("invalid catalog configuration file")
			}
			data, err := io.ReadAll(io.LimitReader(file, postgres.CatalogRequestMaxBytes+1))
			if err != nil {
				return errors.New("cannot read catalog configuration")
			}
			request, err := postgres.DecodeCatalogRequest(data)
			if err != nil {
				return err
			}
			// Do not let libpq environment defaults supply options, alternate TLS
			// material or credentials absent from the approved review.
			for _, entry := range os.Environ() {
				if strings.HasPrefix(entry, "PG") {
					return errors.New("catalog requires an isolated PostgreSQL environment")
				}
			}
			ca := os.Getenv("SSL_CERT_FILE")
			if ca != "" {
				expected, err := filepath.Abs(filepath.Join(filepath.Dir(args[0]), "source-ca.pem"))
				if err != nil || ca != expected {
					return errors.New("catalog source CA is not operation-bound")
				}
				caFile, err := os.Open(ca)
				if err != nil {
					return errors.New("reviewed catalog source CA is unavailable")
				}
				caData, readErr := io.ReadAll(io.LimitReader(caFile, (64<<10)+1))
				_ = caFile.Close()
				digest := sha256.Sum256(caData)
				if readErr != nil || len(caData) > 64<<10 || hex.EncodeToString(digest[:]) != request.SourceCASHA256 {
					return errors.New("catalog source CA changed after review")
				}
			} else if request.SourceCASHA256 != "" {
				return errors.New("reviewed catalog source CA is missing")
			}
			dsn := os.Getenv("AGEFREIGHTER_SOURCE_DSN")
			if err := postgres.ValidateCatalogConnection(request, dsn, ca); err != nil {
				return err
			}
			doc, err := postgres.ReadCatalog(command.Context(), dsn, request.Schemas)
			if err != nil {
				return err
			}
			return json.NewEncoder(command.OutOrStdout()).Encode(doc)
		},
	}
}
