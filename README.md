# LangChain Agent Lab

A demonstration app using LangChain, Anthropic, and Tavily search tools.

## What does this app do?

This app demonstrates how to build an intelligent conversational agent using the LangChain framework. The agent leverages Anthropic's Claude model for natural language understanding and generation, and integrates the Tavily search tool to answer questions with up-to-date web search results.

When you run the app, it:
- Accepts a user message (e.g., "hi im bob! and i live in sf").
- Passes the message to the agent, which uses the Claude model to interpret and respond.
- If the agent determines that a web search is needed, it uses the Tavily search tool to fetch relevant information from the internet.
- Maintains conversational context using in-memory checkpointing, so the agent can remember previous interactions within a session.
- Streams the agent's responses to your terminal, showing how the agent reasons and responds step by step.

This setup is ideal for experimenting with conversational AI, inte

## Requirements

- Python 3.8+
- `pip`
- API keys for Anthropic and Tavily

## Setup

1. **Clone the repository:**  
	```
	git clone https://github.com/akaf47/langchain-agent-lab
	```
2. **Create a virtual environment:** 		 
	```
	python -m venv .venv  
	```
3. **Activate the virtual environment:**  
	```
	source .venv/bin/activate  
	```
    On Windows:  
    ```
	.venv\Scripts\activate
	``` 

4. **Install dependencies:**  
	```
	pip install -r requirements.txt
	```

5. **Set up your `.env` file:**  
- Create a file named `.env` in the root directory. 
- Add your API keys for Anthropic and Tavily in the following format:
	 ```
	 ANTHROPIC_API_KEY=your_anthropic_api_key
	 TAVILY_API_KEY=your_tavily_api_key
	 ```
6. **Running the App:**  
   Run the app using the command:  
   ```
   python langchain_agent_lab.py
   ```  
   You should see the agent's output in your terminal.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
