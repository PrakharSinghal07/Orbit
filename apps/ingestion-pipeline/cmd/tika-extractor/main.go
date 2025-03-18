package main

import (
	"fmt"
	"log"

	"github.com/NEMYSESx/orbit/internal/tika"
)

const (
	tikaServerURL = "http://localhost:9998"
	pdfDir        = "./data/pdfs"
	outputFile    = "extracted_texts.json"
	batchSize     = 5
)

func main() {
	client := tika.NewClient(tikaServerURL)
	processor := tika.NewProcessor(client)
	batchProcessor := tika.NewBatchProcessor(processor, batchSize)

	fmt.Println("🔄 Starting batch PDF extraction...")

	// Process PDFs in batch
	extractedTexts, err := batchProcessor.ProcessDirectory(pdfDir)
	if err != nil {
		log.Fatalf("Error processing PDFs: %v", err)
	}

	// Save results to JSON for Airbyte ingestion
	if err := tika.SaveToJSON(outputFile, extractedTexts); err != nil {
		log.Fatalf("Error saving to JSON: %v", err)
	}

	fmt.Printf("✅ Extraction complete. Data saved to %s\n", outputFile)
}
