package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/NEMYSESx/orbit/apps/data/logs"
)

func main() {
	// Create a custom configuration
	config := logs.DefaultLogConfig()
	
	// Customize configuration if needed
	config.OutputDir = "./synthetic_logs"
	config.LogsPerMinute = map[string]int{
		"system":  20,  // 20 logs per minute (1 every 3 seconds)
		"network": 10,  // 10 logs per minute (1 every 6 seconds)
		"cluster": 5,   // 5 logs per minute (1 every 12 seconds)
		"slurm":   15,  // 15 logs per minute (1 every 4 seconds)
	}
	config.RotationInterval = 1 * time.Hour  // Rotate files hourly
	
	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	// Start log generation in a goroutine
	go logs.LogsWithConfig(config)
	
	fmt.Println("Log generator started. Press Ctrl+C to stop.")
	
	// Wait for termination signal
	<-sigChan
	fmt.Println("\nShutting down log generator...")
}