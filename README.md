# 🏥 Healthcare AI Research Agent

A production-ready LLM/RAG application for healthcare market intelligence and research. Built with LangChain, OpenAI API, and Streamlit.

## 📋 Overview

This application enables healthcare professionals to conduct comprehensive market research with full traceability, evidence-based findings, and human approval gates. It addresses the critical challenges of:

- **Fabricated evidence** - All claims are bound to verifiable sources
- **Inference laundering** - Clear distinction between facts, inferences, and recommendations
- **Coverage gaps** - Explicit identification of unestablished information
- **Contradiction detection** - Surfacing conflicting evidence rather than silent resolution

## 🚀 Features

### Core Capabilities
- ✅ **Natural Language Research Requests** - Accepts ambiguous healthcare research queries
- ✅ **Automated Research Planning** - Decomposes requests into objectives and sub-questions
- ✅ **Tool-Based Evidence Collection** - Uses multiple tools for information gathering
- ✅ **Structured Output** - JSON schema-validated reports with full traceability
- ✅ **Human Approval Gate** - Explicit approval required before publication
- ✅ **Quality Control** - Automated detection of unsupported claims and contradictions

### Technical Implementation
- ✅ **LangChain Integration** - Modern LLM orchestration framework
- ✅ **ChatOpenAI Support** - Full OpenAI API integration
- ✅ **Dual Caching System** - In-Memory and SQLite caching with toggle
- ✅ **Streaming Output** - Real-time response generation
- ✅ **Structured JSON Output** - Schema validation and fault-tolerant parsing
- ✅ **Error Handling** - Robust exception management and retry logic

### UI/UX Features
- ✅ **Dark Theme** - Professional dark mode interface
- ✅ **Responsive Design** - Works on desktop and mobile
- ✅ **Card-Based Layout** - Organized information presentation
- ✅ **Urgency Badges** - Color-coded urgency indicators (LOW, MEDIUM, HIGH, EMERGENCY)
- ✅ **Expandable Sections** - Collapsible content for better organization
- ✅ **Tabbed Navigation** - Multi-tab interface for workflow management

## 📦 Installation

### Prerequisites
- Python 3.9+
- OpenAI API Key
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/healthcare-ai-research-agent.git
cd healthcare-ai-research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env