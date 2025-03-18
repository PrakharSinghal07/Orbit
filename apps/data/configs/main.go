package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
)

func generateNetworkConfigs(outputDir string) error {
	netDir := filepath.Join(outputDir, "network")
	if err := os.MkdirAll(netDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create network directory: %v", err)
	}

	baseNetwork := "192.168.1.0/24"
	_, ipNet, err := net.ParseCIDR(baseNetwork)
	if err != nil {
		return fmt.Errorf("failed to parse CIDR: %v", err)
	}

	hosts := getHostsInNetwork(ipNet)
	if len(hosts) < 11 {
		return fmt.Errorf("not enough host IPs in the network")
	}

	dnsConfig := fmt.Sprintf(`# /etc/resolv.conf
# Generated for HPE customer environment

domain customer.local
search customer.local hpe.local

# Primary DNS server
nameserver %s
# Secondary DNS server
nameserver %s
# Tertiary DNS server
nameserver 8.8.8.8

options timeout:2 attempts:3 rotate
`, hosts[1].String(), hosts[2].String())

	ntpConfig := fmt.Sprintf(`# /etc/ntp.conf
# Generated for HPE customer environment

driftfile /var/lib/ntp/drift
logfile /var/log/ntp.log

# Access control
restrict default kod nomodify notrap nopeer noquery
restrict -6 default kod nomodify notrap nopeer noquery
restrict 127.0.0.1
restrict ::1

# NTP servers
server %s iburst
server %s iburst
server %s iburst
server 0.pool.ntp.org iburst

# Local clock
fudge 127.127.1.0 stratum 10

# Key file
keys /etc/ntp/keys
`, hosts[5].String(), hosts[6].String(), hosts[7].String())

	interfaceConfig := fmt.Sprintf(`# /etc/sysconfig/network/ifcfg-eth0
# Generated for HPE customer environment

BOOTPROTO='static'
IPADDR='%s/24'
NAME='Ethernet Connection'
ONBOOT='yes'
STARTMODE='auto'
TYPE='Ethernet'
USERCONTROL='no'
GATEWAY='%s'
`, hosts[10].String(), hosts[1].String())

	if err := os.WriteFile(filepath.Join(netDir, "resolv.conf"), []byte(dnsConfig), 0644); err != nil {
		return fmt.Errorf("failed to write resolv.conf: %v", err)
	}

	if err := os.WriteFile(filepath.Join(netDir, "ntp.conf"), []byte(ntpConfig), 0644); err != nil {
		return fmt.Errorf("failed to write ntp.conf: %v", err)
	}

	if err := os.WriteFile(filepath.Join(netDir, "ifcfg-eth0"), []byte(interfaceConfig), 0644); err != nil {
		return fmt.Errorf("failed to write ifcfg-eth0: %v", err)
	}

	badDns := strings.Replace(dnsConfig, "8.8.8.8", "8.8.8.800", 1) 
	if err := os.WriteFile(filepath.Join(netDir, "resolv.conf.problematic"), []byte(badDns), 0644); err != nil {
		return fmt.Errorf("failed to write problematic resolv.conf: %v", err)
	}

	badNtp := strings.Replace(ntpConfig, "iburst", "burst", -1) 
	badNtp = strings.Replace(badNtp, "restrict default", "# restrict default", 1) 
	if err := os.WriteFile(filepath.Join(netDir, "ntp.conf.problematic"), []byte(badNtp), 0644); err != nil {
		return fmt.Errorf("failed to write problematic ntp.conf: %v", err)
	}

	return nil
}

func generateHaClusterConfigs(outputDir string) error {
	clusterDir := filepath.Join(outputDir, "ha_cluster")
	if err := os.MkdirAll(clusterDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create cluster directory: %v", err)
	}

	corosyncConf := `# /etc/corosync/corosync.conf
# Generated for HPE customer environment

totem {
    version: 2
    secauth: on
    crypto_hash: sha1
    crypto_cipher: aes256
    cluster_name: hpe_cluster
    transport: udpu
    interface {
        ringnumber: 0
        bindnetaddr: 192.168.1.0
        mcastport: 5405
        ttl: 1
    }
}

nodelist {
    node {
        ring0_addr: 192.168.1.101
        nodeid: 1
    }
    node {
        ring0_addr: 192.168.1.102
        nodeid: 2
    }
    node {
        ring0_addr: 192.168.1.103
        nodeid: 3
    }
}

quorum {
    provider: corosync_votequorum
    expected_votes: 3
    two_node: 0
}

logging {
    fileline: off
    to_logfile: yes
    to_syslog: yes
    logfile: /var/log/cluster/corosync.log
    debug: off
    timestamp: on
    logger_subsys {
        subsys: QUORUM
        debug: off
    }
}
`

	crmConfig := `# Cluster CRM configuration
# Generated for HPE customer environment

node 1: node1
node 2: node2
node 3: node3

primitive ip_resource IPaddr2 \
    params ip=192.168.1.100 cidr_netmask=24 \
    op monitor interval=10s timeout=20s

primitive webserver apache \
    params configfile=/etc/apache2/httpd.conf \
    op monitor interval=10s timeout=20s \
    op start interval=0 timeout=40s \
    op stop interval=0 timeout=60s

primitive db_resource mysql \
    params config=/etc/my.cnf datadir=/var/lib/mysql \
    op monitor interval=20s timeout=30s \
    op start interval=0 timeout=120s \
    op stop interval=0 timeout=120s

group service_group ip_resource webserver db_resource

location loc_prefer_node1 service_group 100: node1
location loc_prefer_node2 service_group 50: node2
location loc_prefer_node3 service_group 25: node3

property stonith-enabled=true
property no-quorum-policy=stop
property default-resource-stickiness=100
`

	if err := os.WriteFile(filepath.Join(clusterDir, "corosync.conf"), []byte(corosyncConf), 0644); err != nil {
		return fmt.Errorf("failed to write corosync.conf: %v", err)
	}

	if err := os.WriteFile(filepath.Join(clusterDir, "crm_config"), []byte(crmConfig), 0644); err != nil {
		return fmt.Errorf("failed to write crm_config: %v", err)
	}

	return nil
}

func getHostsInNetwork(ipNet *net.IPNet) []net.IP {
	var ips []net.IP
	
	ip := ipNet.IP.To4()
	mask := ipNet.Mask
	
	ones, bits := mask.Size()
	hostBits := bits - ones
	numHosts := 1<<uint(hostBits) - 2 
	
	for i := 1; i <= numHosts; i++ {
		hostIP := make(net.IP, len(ip))
		copy(hostIP, ip)
		for j := 3; j >= 0; j-- {
			hostIP[j] += byte(i % 256)
			i = i / 256
		}
		
		ips = append(ips, hostIP)
	}
	
	return ips
}

func main() {
	outputDir := "./synthetic_configs"
	
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		os.Exit(1)
	}
	
	if err := generateNetworkConfigs(outputDir); err != nil {
		fmt.Printf("Error generating network configs: %v\n", err)
		os.Exit(1)
	}
	
	if err := generateHaClusterConfigs(outputDir); err != nil {
		fmt.Printf("Error generating HA cluster configs: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Printf("Generated configuration files in %s\n", outputDir)
}