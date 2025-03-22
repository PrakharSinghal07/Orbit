package tika

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
)

type ExtractedData struct {
	FileName string `json:"file_name"`
	Text     string `json:"text"`
}

type Processor struct {
	Client *Client
}

func NewProcessor(client *Client) *Processor {
	return &Processor{Client: client}
}

func (p *Processor) ExtractFromFile(filePath string) (*ExtractedData, error) {
	pdfData, err := os.ReadFile(filePath)
	if err != nil {
		log.Printf("Failed to read file %s: %v", filePath, err)
		return nil, err
	}

	text, err := p.Client.ExtractText(pdfData)
	if err != nil {
		log.Printf("Error extracting text from %s: %v", filePath, err)
		return nil, err
	}

	return &ExtractedData{FileName: filepath.Base(filePath), Text: text}, nil
}


func SaveToJSON(outputFile string, data []ExtractedData) error {
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(outputFile, jsonData, 0644)
}

func SaveSingleToJSON(outputFile string, data ExtractedData) error {
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(outputFile, jsonData, 0644)
}