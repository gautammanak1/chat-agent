![uagents](https://img.shields.io/badge/uagents-4A90E2)  ![innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3) ![chatprotocol](https://img.shields.io/badge/chatprotocol-1D3BD4) [![X](https://img.shields.io/badge/X-black.svg?logo=X&logoColor=white)](https://x.com/gautammanak02) ![tag:spotlight](https://img.shields.io/badge/spotlight-0EA5E9)

---


## 📊 Chart Generator Agen

Transform your data visualization needs into beautiful charts with just a simple chat message. The Chart Generator Agent leverages ASI 1 LLM to generate Python code and creates stunning visualizations using matplotlib, all executed securely and displayed directly in your chat interface.

### What it Does

The Chart Generator Agent acts as your personal chart creation assistant, using advanced AI to understand your visualization requirements and generate custom Python code. It executes the code locally, captures the charts, uploads them to Cloudinary, and displays them directly in chat messages using markdown formatting.

## ✨ Key Features

*   **AI-Powered Code Generation**: Uses ASI 1 LLM to generate optimized Python matplotlib code based on natural language descriptions.
*   **Multiple Chart Types**: Supports line charts, bar charts, pie charts, scatter plots, box plots, and any custom matplotlib visualization.
*   **Secure Local Execution**: Charts are generated locally using matplotlib with Agg backend for safe, non-interactive execution.
*   **Cloud Storage**: Automatic upload to Cloudinary ensures your charts are accessible via permanent URLs.
*   **Chat Integration**: Built on uAgents chat protocol for seamless integration with agent-to-agent communication.
*   **Zero Local Storage**: Charts are captured directly to memory and uploaded to Cloudinary - no local files are saved.
*   **Smart Code Extraction**: Automatically extracts Python code from LLM responses, handling various code block formats.
*   **Error Handling**: Comprehensive error handling with informative messages for debugging.

## 🚀 Example Usage

Once deployed, you can interact with the Chart Generator Agent by sending chart requests via uAgent chat protocol.

### Example Query

```
"Create an interactive dashboard layout with three linked charts: a pie chart of product categories, a bar chart of revenue by region, and a line chart of monthly profit. Selecting a region in the bar chart should filter the other two charts automatically."
```

### Expected Output

The agent will respond with:

![image](https://res.cloudinary.com/doesqlfyi/image/upload/v1762378904/charts/chart_1762378903_1.png)



The agent will start and publish its manifest to Agentverse. You'll see the agent's address printed in the console.


## 🏗️ Architecture

---

## 🧠 Inspired by

*   [Fetch.ai uAgents](https://github.com/fetchai/uAgents)
*   [ASI:One](https://asi1.ai/chat)
