package tika

import (
	"log"
	"os"
	"path/filepath"
	"sync"
)

// BatchProcessor handles batch processing of PDFs.
type BatchProcessor struct {
	Processor *Processor
	BatchSize int
}

// NewBatchProcessor initializes a BatchProcessor.
func NewBatchProcessor(processor *Processor, batchSize int) *BatchProcessor {
	return &BatchProcessor{Processor: processor, BatchSize: batchSize}
}

// ProcessDirectory scans a directory and processes PDFs in batches.
func (bp *BatchProcessor) ProcessDirectory(pdfDir string) ([]ExtractedData, error) {
	files, err := bp.getPDFFiles(pdfDir)
	if err != nil {
		return nil, err
	}

	return bp.processInBatch(files)
}

// getPDFFiles retrieves all PDF files in the directory.
func (bp *BatchProcessor) getPDFFiles(dir string) ([]string, error) {
	var pdfFiles []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && filepath.Ext(info.Name()) == ".pdf" {
			pdfFiles = append(pdfFiles, path)
		}
		return nil
	})
	return pdfFiles, err
}

// processInBatch runs concurrent goroutines for processing PDFs.
func (bp *BatchProcessor) processInBatch(files []string) ([]ExtractedData, error) {
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, bp.BatchSize)
	results := make(chan ExtractedData, len(files))

	for _, file := range files {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(pdfFile string) {
			defer wg.Done()
			data, err := bp.Processor.ExtractFromFile(pdfFile)
			if err == nil {
				results <- *data
			} else {
				log.Printf("Failed to process file: %s", pdfFile)
			}
			<-semaphore
		}(file)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var extractedData []ExtractedData
	for result := range results {
		extractedData = append(extractedData, result)
	}

	return extractedData, nil
}
