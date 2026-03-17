# Scientific Hub: AI Research Assistant

## Overview
**Scientific Hub** is an advanced AI research assistant designed to automate the tedious processes of literature review, gap analysis, and grant proposal synthesis. By utilizing a multi-agent system powered by CrewAI, it orchestrates specialized AI agents to simulate a real research team. This drastically reduces the time taken from ideation to obtaining a highly novel, structured proposal skeleton.

## Core Features
- **Model Agnostic**: Seamlessly switch between OpenAI, Groq, Gemini, and local Ollama models.
- **Dynamic Speed Modes**:
  - ⚡⚡ **Flash**: 2 agents, ultra-brief output (~1 min for rapid ideation).
  - ⚡ **Turbo**: Complete 6-agent structured flow (~5 min).
  - 🔬 **Standard**: High-detail, deep reasoning mode.
- **Real-time Agent Telemetry**: Output parsing to watch the AI reason live.
- **Research Analytics**: Includes insights into novelty, topic analysis, and more.

## The 6-Agent AI Pipeline
The project utilizes a powerful pipeline consisting of the following specialized AI agents:
1. 📚 **Literature Mining**: Scans modern databases for core papers.
2. 📈 **Trend Analysis**: Identifies temporal themes based on literature.
3. 🔍 **Gap Identification**: Pinpoints unexplored or undertheorized areas.
4. 🔬 **Methodology Design**: Structures a concrete research plan and dataset strategy.
5. ✍️ **Grant Writing**: Generates a 5-section persuasive grant outline.
6. ✅ **Novelty Audit**: Verifies originality and provides a rigid Novelty Score.

## Technical Architecture
- **Frontend**: Built with [Streamlit](https://streamlit.io/) in Python, featuring a custom dark 'IDE / Hacker' scientific vibe.
- **Agent Orchestration**: Powered by [CrewAI Framework](https://www.crewai.com/).
- **LLM Engine**: Utilizes `litellm` and direct API SDKs handling inference asynchronously.
- **Data Handling**: Regex-based output parsing to separate technical audits from proposal text.
- **Background Execution**: Python threading ensures the user interface remains responsive during long generation processes.

## Setup & Installation

### 1. Requirements
Ensure you have Python installed. You can install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Rename the provided `.env.example` file to `.env` and configure your API keys:
```
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```
*(Leave keys blank if you intend to use local `ollama`)*

### 3. Running the Application
You can run the Streamlit application using the provided script:
```bash
# On Windows
run_app.bat

# Or directly using Streamlit
streamlit run app.py
```

## Creating a Presentation
You can also generate an automatic presentation summarizing the platform by running:
```bash
python generate_presentation.py
```
This will generate a `Scientific_Hub_Presentation.pptx` file.

## Why Scientific Hub?
Scientific Hub reimagines how academics interact with the endless sea of literature. It turns days of manual searching and summarizing into a 5-minute automated pipeline. As a result, researchers can spend more time on actual scientific innovation and less time on repetitive drafting.
