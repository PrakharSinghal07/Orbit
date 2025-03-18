package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
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

func generateLogs(logType string, numEntries int, startDate, endDate time.Time, includeErrors bool) []string {
	logs := make([]string, 0, numEntries)
	
	errorCount := 0
	warningCount := 0
	if includeErrors {
		errorCount = int(float64(numEntries) * 0.15)
		warningCount = int(float64(numEntries) * 0.25)
	}
	
	timestamps := make([]time.Time, numEntries)
	for i := 0; i < numEntries; i++ {
		timestamps[i] = generateTimestamp(startDate, endDate)
	}
	sort.Slice(timestamps, func(i, j int) bool {
		return timestamps[i].Before(timestamps[j])
	})
	
	for i, timestamp := range timestamps {
		levelIndex := 0
		if i < errorCount {
			levelIndex = 1 
		} else if i < errorCount+warningCount {
			levelIndex = 2 
		} else {
			if len(LogTypes[logType]) > 4 {
				levelIndex = []int{0, 3, 4}[currentRand.Intn(3)]
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
		logs = append(logs, fmt.Sprintf("%s %s", timestampStr, logEntry))
	}
	
	return logs
}

func writeLogsToFile(logs []string, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	for _, log := range logs {
		_, err := file.WriteString(log + "\n")
		if err != nil {
			return err
		}
	}
	
	return nil
}

func generateAllLogs(outputDir string) error {
	err := os.MkdirAll(outputDir, os.ModePerm)
	if err != nil {
		return err
	}
	
	endDate := time.Now()
	startDate := endDate.AddDate(0, 0, -30)
	
	for logType := range LogTypes {
		numEntries := 500
		if logType == "system" {
			numEntries = 1000
		}
		
		logs := generateLogs(logType, numEntries, startDate, endDate, true)
		err := writeLogsToFile(logs, filepath.Join(outputDir, logType+".log"))
		if err != nil {
			return err
		}
	}
	
	fmt.Printf("Generated logs in %s\n", outputDir)
	return nil
}

func main() {
	outputDir := "./synthetic_logs"
	err := generateAllLogs(outputDir)
	if err != nil {
		fmt.Printf("Error generating logs: %v\n", err)
		os.Exit(1)
	}
}