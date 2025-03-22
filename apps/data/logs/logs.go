package logs

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

var LogTypes = map[string][]string{
	"system": {
		"NOTICE: System {service} {action} at {timestamp}",
		"ERROR: Failed to {action} {service} - {error_code}: {error_message}",
		"WARNING: {service} resource utilization at {percentage}%",
		"INFO: Configuration change detected for {service}",
		"CRITICAL: {service} is not responding - attempting restart",
	},
	"network": {
		"INFO: Interface {interface} {status} at {timestamp}",
		"ERROR: DNS resolution failed for {hostname} - {error_code}",
		"WARNING: High packet loss on {interface}: {percentage}%",
		"NOTICE: NTP synchronization {status} with server {ntp_server}",
		"ERROR: Cannot establish connection to {hostname}:{port} - {error_message}",
	},
	"cluster": {
		"INFO: Node {node} joined cluster at {timestamp}",
		"WARNING: Possible split-brain detected between nodes {node1} and {node2}",
		"ERROR: Resource {resource} failed on node {node} - {error_message}",
		"NOTICE: Performing failover of {resource} from {node1} to {node2}",
		"CRITICAL: Cluster partition detected - {partition_details}",
	},
	"slurm": {
		"INFO: Job {job_id} submitted by {user} at {timestamp}",
		"ERROR: Job {job_id} failed on node {node} - {error_message}",
		"WARNING: Node {node} approaching memory limit: {percentage}%",
		"NOTICE: Scheduler configuration reloaded at {timestamp}",
		"INFO: Reservation {reservation_id} created for {time_period}",
	},
}

var (
	Services      = []string{"httpd", "sshd", "mariadb", "nfs", "ldap", "cron", "pacemaker", "corosync"}
	Actions       = []string{"started", "stopped", "restarted", "failed", "configured"}
	ErrorCodes    = []string{"E1001", "E1002", "E1003", "E1004", "E1005"}
	ErrorMessages = []string{
		"insufficient permissions",
		"resource unavailable",
		"timeout exceeded",
		"dependency failed",
		"invalid configuration",
		"network unreachable",
		"authentication failed",
	}
	Interfaces  = []string{"eth0", "eth1", "bond0", "ib0", "ens192", "ens256"}
	Statuses    = []string{"up", "down", "degraded", "flapping", "reconfigured"}
	Hostnames   = []string{"compute01", "compute02", "mgmt01", "storage01", "headnode", "login01"}
	NtpServers  = []string{"0.pool.ntp.org", "1.pool.ntp.org", "timeserver.local", "ntp.hpe.com"}
	Nodes       = []string{"node001", "node002", "node003", "node004", "node005"}
	Resources   = []string{"virtual_ip", "shared_fs", "database", "web_service", "scheduler"}
	Users       = []string{"admin", "user1", "operator", "scheduler", "root"}
	currentRand = rand.New(rand.NewSource(time.Now().UnixNano()))
)

type LogConfig struct {
	OutputDir          string
	RotationInterval   time.Duration
	LogsPerMinute      map[string]int
	IncludeErrors      bool
	RotateFilesBySize  bool
	MaxFileSizeMB      int
	RetentionPeriod    time.Duration
}

func DefaultLogConfig() LogConfig {
	return LogConfig{
		OutputDir:        "./synthetic_logs",
		RotationInterval: 24 * time.Hour,
		LogsPerMinute: map[string]int{
			"system":  10,
			"network": 5,
			"cluster": 3,
			"slurm":   8,
		},
		IncludeErrors:     true,
		RotateFilesBySize: false,
		MaxFileSizeMB:     100,
		RetentionPeriod:   30 * 24 * time.Hour, 
	}
}

func generateTimestamp(startDate, endDate time.Time) time.Time {
	delta := endDate.Sub(startDate)
	deltaNanoseconds := int64(delta)
	randomDelta := currentRand.Int63n(deltaNanoseconds)
	return startDate.Add(time.Duration(randomDelta))
}

func getRandomElement(slice []string) string {
	return slice[currentRand.Intn(len(slice))]
}

func replacePlaceholders(template string, values map[string]string) string {
	result := template
	for key, value := range values {
		result = strings.Replace(result, "{"+key+"}", value, -1)
	}
	return result
}

func generateLogEntry(logType string, timestamp time.Time, includeErrors bool) string {
	levelIndex := 0
	
	if includeErrors {
		roll := currentRand.Float64()
		if roll < 0.15 {
			levelIndex = 1 
		} else if roll < 0.40 {
			levelIndex = 2 
		} else {
			if len(LogTypes[logType]) > 4 {
				levelIndex = []int{0, 3, 4}[currentRand.Intn(3)]
			}
		}
	}
	
	template := LogTypes[logType][levelIndex%len(LogTypes[logType])]
	
	timestampStr := timestamp.Format("2006-01-02 15:04:05.000")
	
	values := map[string]string{
		"timestamp":        timestampStr,
		"service":          getRandomElement(Services),
		"action":           getRandomElement(Actions),
		"error_code":       getRandomElement(ErrorCodes),
		"error_message":    getRandomElement(ErrorMessages),
		"percentage":       fmt.Sprintf("%d", currentRand.Intn(20)+80),
		"interface":        getRandomElement(Interfaces),
		"status":           getRandomElement(Statuses),
		"hostname":         getRandomElement(Hostnames),
		"port":             fmt.Sprintf("%d", currentRand.Intn(64511)+1024),
		"ntp_server":       getRandomElement(NtpServers),
		"node":             getRandomElement(Nodes),
		"node1":            getRandomElement(Nodes),
		"node2":            getRandomElement(Nodes),
		"resource":         getRandomElement(Resources),
		"partition_details": fmt.Sprintf("nodes %s,%s isolated", getRandomElement(Nodes), getRandomElement(Nodes)),
		"job_id":           fmt.Sprintf("%d", currentRand.Intn(9000)+1000),
		"user":             getRandomElement(Users),
		"reservation_id":   fmt.Sprintf("res_%d", currentRand.Intn(900)+100),
		"time_period":      fmt.Sprintf("%d hours", currentRand.Intn(24)+1),
	}
	
	logEntry := replacePlaceholders(template, values)
	return fmt.Sprintf("%s %s", timestampStr, logEntry)
}

func appendLogToFile(logEntry string, filename string) error {
	_, err := os.Stat(filename)
	if os.IsNotExist(err) {
		dir := filepath.Dir(filename)
		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			return err
		}
		
		file, err := os.Create(filename)
		if err != nil {
			return err
		}
		defer file.Close()
	}
	
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	
	_, err = file.WriteString(logEntry + "\n")
	return err
}

func rotateLogFile(logType string, config LogConfig) (string, error) {
	timestamp := time.Now().Format("20060102-150405")
	baseDir := filepath.Join(config.OutputDir, logType)
	
	if err := os.MkdirAll(baseDir, os.ModePerm); err != nil {
		return "", err
	}
	
	newFilename := filepath.Join(baseDir, fmt.Sprintf("%s-%s.log", logType, timestamp))
	return newFilename, nil
}

func cleanupOldLogs(config LogConfig) error {
	cutoffTime := time.Now().Add(-config.RetentionPeriod)
	
	return filepath.Walk(config.OutputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if info.IsDir() {
			return nil
		}
		
		if strings.HasSuffix(info.Name(), ".log") && info.ModTime().Before(cutoffTime) {
			return os.Remove(path)
		}
		
		return nil
	})
}

func StartContinuousLogging(config LogConfig) {
	fmt.Printf("Starting continuous log generation in %s\n", config.OutputDir)
	
	if err := os.MkdirAll(config.OutputDir, os.ModePerm); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		return
	}
	
	currentLogFiles := make(map[string]string)
	for logType := range config.LogsPerMinute {
		newFile, err := rotateLogFile(logType, config)
		if err != nil {
			fmt.Printf("Error creating log file for %s: %v\n", logType, err)
			return
		}
		currentLogFiles[logType] = newFile
		fmt.Printf("Created initial log file for %s: %s\n", logType, newFile)
	}
	
	rotationTicker := time.NewTicker(config.RotationInterval)
	defer rotationTicker.Stop()
	
	cleanupTicker := time.NewTicker(24 * time.Hour)
	defer cleanupTicker.Stop()
	
	logIntervals := make(map[string]time.Duration)
	for logType, logsPerMinute := range config.LogsPerMinute {
		if logsPerMinute <= 0 {
			continue
		}
		interval := time.Minute / time.Duration(logsPerMinute)
		logIntervals[logType] = interval
	}
	
	logTickers := make(map[string]*time.Ticker)
	for logType, interval := range logIntervals {
		logTickers[logType] = time.NewTicker(interval)
	}
	
	defer func() {
		for _, ticker := range logTickers {
			ticker.Stop()
		}
	}()
	
	fileSizes := make(map[string]int64)
	

	done := make(chan bool)
	
	for logType, ticker := range logTickers {
		go func(lt string, tk *time.Ticker) {
			for {
				select {
				case <-tk.C:
					logEntry := generateLogEntry(lt, time.Now(), config.IncludeErrors)
					
				
					err := appendLogToFile(logEntry, currentLogFiles[lt])
					if err != nil {
						fmt.Printf("Error writing to log file for %s: %v\n", lt, err)
						continue
					}
					
					if config.RotateFilesBySize {
						fileSizes[lt] += int64(len(logEntry) + 1) 
						
						if fileSizes[lt] > int64(config.MaxFileSizeMB)*1024*1024 {
							newFile, err := rotateLogFile(lt, config)
							if err != nil {
								fmt.Printf("Error rotating log file for %s: %v\n", lt, err)
								continue
							}
							
							fmt.Printf("Rotated log file for %s due to size: %s\n", lt, newFile)
							currentLogFiles[lt] = newFile
							fileSizes[lt] = 0
						}
					}
					
				case <-done:
					return
				}
			}
		}(logType, ticker)
	}
	
	for {
		select {
		case <-rotationTicker.C:
			for logType := range config.LogsPerMinute {
				newFile, err := rotateLogFile(logType, config)
				if err != nil {
					fmt.Printf("Error rotating log file for %s: %v\n", logType, err)
					continue
				}
				
				fmt.Printf("Rotated log file for %s at scheduled interval: %s\n", logType, newFile)
				currentLogFiles[logType] = newFile
				fileSizes[logType] = 0
			}
			
		case <-cleanupTicker.C:
			if err := cleanupOldLogs(config); err != nil {
				fmt.Printf("Error cleaning up old log files: %v\n", err)
			} else {
				fmt.Println("Cleaned up old log files")
			}
			
		case <-done:
			return
		}
	}
}

func Logs() {
	config := DefaultLogConfig()
	
	
	StartContinuousLogging(config)
}

func LogsWithConfig(config LogConfig) {
	StartContinuousLogging(config)
}