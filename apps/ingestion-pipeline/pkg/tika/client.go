package tika

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
)

type Client struct {
	TikaURL string
}

func NewClient(tikaURL string) *Client {
	return &Client{TikaURL: tikaURL}
}

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
		return "", fmt.Errorf("tika server returned status: %d", resp.StatusCode)
	}

	text, err := io.ReadAll(resp.Body)
	return string(text), err
}