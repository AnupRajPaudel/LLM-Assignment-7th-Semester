# Assignment 3.2: Multi-Agent LLM System

## Overview
This assignment involves building a multi-agent framework where multiple LLM agents collaborate to solve complex tasks. Each agent is specialized for a particular function:

- **Planning Agent**: Decomposes complex queries into smaller, manageable subtasks  
- **Summarization Agent**: Produces concise summaries of lengthy texts and documents  
- **QA Agent**: Provides detailed answers to specific questions  
- **Coordinator Agent**: Manages communication and coordination among agents

## System Architecture
The system leverages message passing for inter-agent communication alongside a shared memory module that stores intermediate data and context for smooth collaboration.

## Key Features
- Modular design with agents focusing on distinct tasks  
- Communication handled via a message passing protocol  
- Shared memory to keep track of conversation context and intermediate results  
- Maintains conversation history for continuity  
- Supports task delegation and integration of agent outputs


## Directory Structure
```
Assignment-3.2-multi-agent-system/
├── README.md
├── requirements.txt
├── multi_agent_system.ipynb          # Main notebook with examples
├── main.py                           # Command-line interface
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                 # Base agent class
│   ├── planning_agent.py             # Task planning agent
│   ├── summarization_agent.py        # Text summarization agent
│   ├── qa_agent.py                   # Question answering agent
│   └── coordinator_agent.py          # System coordinator
├── utils/
│   ├── __init__.py
│   ├── message_system.py             # Message passing system
│   ├── shared_memory.py              # Shared memory implementation
│   └── llm_interface.py              # LLM API interface
└── data/
    └── sample_documents.json         # Sample data for testing
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
Open `multi_agent_system.ipynb` to see interactive examples and demonstrations.

### Command Line
```bash
python main.py "What are the key insights from the attached documents and how should we proceed?"
```

## Example Scenarios
1. **Document Analysis**: Upload multiple documents, get summaries and key insights
2. **Research Planning**: Break down complex research questions into actionable steps
3. **Multi-step QA**: Answer complex questions requiring multiple reasoning steps

## Configuration
Set your OpenAI API key in environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Architecture Details
- Agents communicate through structured messages
- Shared memory maintains conversation context
- Coordinator routes messages and aggregates results
- Each agent has specialized prompts and capabilities
