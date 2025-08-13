# Orbit - Chatbot Project

Orbit is a chatbot application that aims to assist service engineers with debugging issues through an interactive interface. 

# Live Link

[Orbit : https://orbit-lcm-hpe.netlify.app/](https://orbit-lcm-hpe.netlify.app/)

This project was developed as part of my **Project Internship** at **Hewlett Packard Enterprise**.  
The full project was a collaborative effort by a 5-member team, with my primary responsibility being the **frontend development** and **chat management** with the FastAPI backend.

You can view my **individual contributions** here: [My Contributions Repository](https://github.com/PrakharSinghal07/Orbit-HPE-CTY)

# Orbit Setup Guide

## Architecture Overview
The application consists of the following services:

- **Apache Tika** – Document text extraction service  
- **Apache Kafka** – Message streaming platform  
- **Ingestion Pipeline** – Processes and ingests documents  
- **Embedding Pipeline** – Generates text embeddings  
- **RAG API** – Retrieval-Augmented Generation service  
- **Chat Backend** – Main application backend  
- **Chat Frontend** – React-based user interface  

## Setup

```bash
git clone https://github.com/NEMYSESx/Orbit
cd Orbit
```

## Build and Start Services

```bash
docker-compose up --build
docker-compose up -d
```

## Demo
 [Download or view the demo video](./Orbit%20demo%20video.mp4)


