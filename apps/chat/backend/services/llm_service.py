import random


suggestion_bank = [
    "Steps to fix a reboot loop",
    "How to check node logs?",
    "What is an HA cluster?",
    "How to resolve configuration mismatches?",
    "Why does my node hang?",
    "What are SLURM scheduler issues?",
    "How to analyze performance degradation?",
]

def get_llm_response(query: str):
    response = f"Mocked response to: '{query}'"
    return response

def getSuggestions():
  suggestions = random.sample(suggestion_bank, 3)
  return suggestions
