package tika_extractor

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/NEMYSESx/orbit/apps/ingestion-pipeline/pkg/tika"
)

const (
	tikaServerURL = "http://localhost:9998"
	pdfDir        = "./apps/data/docs/hpe_docs"
	outputDir     = "./apps/data/output/json"  
	batchSize     = 5
)

func TikaExtractor() {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}

	client := tika.NewClient(tikaServerURL)
	processor := tika.NewProcessor(client)
	batchProcessor := tika.NewBatchProcessor(processor, batchSize)

	fmt.Println("🔄 Starting batch PDF extraction...")

	extractedTexts, err := batchProcessor.ProcessDirectory(pdfDir)
	if err != nil {
		log.Fatalf("Error processing PDFs: %v", err)
	}

	fmt.Printf("Saving %d documents as individual JSON files...\n", len(extractedTexts))
	successCount := 0
	
	for _, data := range extractedTexts {
		outputPath := filepath.Join(outputDir, filepath.Base(data.FileName)+".json")
		if err := tika.SaveSingleToJSON(outputPath, data); err != nil {
			log.Printf("Error saving %s: %v", data.FileName, err)
		} else {
			successCount++
		}
	}

	fmt.Printf("✅ Extraction complete. %d/%d documents saved to %s\n", 
		successCount, len(extractedTexts), outputDir)
}