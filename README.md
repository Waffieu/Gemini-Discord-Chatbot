# Nyxie V2 - Advanced Protogen AI Discord Bot

## Overview
Nyxie V2 is a sophisticated Discord chatbot powered by Google's Gemini AI. It's designed as a Protogen AI with a complex, enigmatic personality capable of engaging in deep conversations, analyzing images, and responding in multiple languages. The bot features a memory system that allows it to remember past interactions with users.

## Features
- **Advanced AI Conversations**: Powered by Google's Gemini 2.0 Flash model for natural and engaging interactions
- **Multilingual Support**: Automatically detects and responds in the user's language
- **Image Analysis**: Can analyze and discuss images shared in Discord
- **Long-term Memory**: Remembers past conversations with users using FAISS vector database
- **Web Search Integration**: Can search the web using DuckDuckGo to provide up-to-date information
- **Rich Personality**: Embodies a complex Protogen AI character with unique traits and communication style

## Installation

### Prerequisites
- Python 3.8 or higher
- A Discord bot token
- A Google Gemini API key

### Setup
1. Clone this repository or download the source code

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your API keys:
   - Create a `config.json` file with the following structure:
     ```json
     {
       "discord_token": "YOUR_DISCORD_BOT_TOKEN",
       "gemini_api_key": "YOUR_GEMINI_API_KEY"
     }
     ```
   - Alternatively, you can use environment variables by creating a `.env` file:
     ```
     DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN
     GEMINI_TOKEN=YOUR_GEMINI_API_KEY
     ```

4. Run the bot:
   ```
   python bot.py
   ```

## Usage
Once the bot is running and added to your Discord server, you can interact with it in any text channel where it has permission to read and send messages.

### Basic Commands
- Simply mention the bot or send a direct message to start a conversation
- The bot will automatically detect your language and respond accordingly
- Share images for the bot to analyze and discuss
- Ask "remember when..." questions to access the bot's memory of past conversations

### Memory System
Nyxie V2 uses a sophisticated memory system based on FAISS (Facebook AI Similarity Search) to store and retrieve memories of past conversations. This allows the bot to maintain context and build relationships with users over time.

## Dependencies
The bot relies on the following Python packages:
- discord.py - Discord API integration
- google-generativeai - Google Gemini AI integration
- faiss-cpu - Vector similarity search for memory system
- sentence-transformers - Text embedding generation
- numpy - Numerical operations
- duckduckgo-search - Web search functionality
- langdetect - Language detection
- requests - HTTP requests
- Pillow - Image processing

## Project Structure
- `bot.py` - Main bot code and logic
- `config.json` - Configuration file for API keys
- `.env` - Alternative environment variables file
- `requirements.txt` - Python dependencies
- `user_memories/` - Directory storing user conversation memories

## Security Note
This bot requires API keys to function. Never share your `config.json` or `.env` file publicly, and ensure they are added to your `.gitignore` if using version control.

## License
This project is available for personal and educational use.