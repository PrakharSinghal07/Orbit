package tika

import (
	"bytes"
	"io"
	"log"
	"net/http"
)

// Client interacts with an Apache Tika server.
type Client struct {
	TikaURL string
}

// NewClient initializes the Tika client.
func NewClient(tikaURL string) *Client {
	return &Client{TikaURL: tikaURL}
}

// ExtractText sends a PDF file to Apache Tika and extracts text.
func (c *Client) ExtractText(pdfData []byte) (string, error) {
	req, err := http.NewRequest("PUT", c.TikaURL+"/tika", bytes.NewReader(pdfData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/pdf")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Tika server returned status: %d", resp.StatusCode)
		return "", err
	}

	text, err := io.ReadAll(resp.Body)
	return string(text), err
}
