# AI-Powered Zapier Calendar Agent

A conversational AI assistant that integrates with Google Calendar through Zapier's MCP (Model Context Protocol) to help you manage your calendar events using natural language.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)  
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [API Information](#api-information)
- [Contributing](#contributing)

## Features

### Core Capabilities
- **AI Intent Analysis**: Automatically understands calendar requests in natural language using OpenAI GPT-4o-mini
- **Calendar Management**: Find events, create meetings, and check availability
- **Conversation Memory**: Remembers context from previous interactions in the same session
- **Rich Console Interface**: Beautiful terminal UI with colors, panels, and formatted output
- **Debug Tools**: Built-in debugging and diagnostic commands for troubleshooting

### Calendar Operations
- **Find Events**: Search for events by date, time, or keywords
- **Create Events**: Add new events using natural language or structured input
- **Check Availability**: Identify busy periods and scheduling conflicts
- **Event Formatting**: Display events with duration, attendee count, and links

## Prerequisites

- **Python 3.8 or higher**
- **OpenAI API key** with access to GPT-4o-mini
- **Zapier account** with Google Calendar integration configured
- **Zapier MCP server URL** for API access
- **Google Calendar** with appropriate permissions

## Installation

### Step 1: Download the Code
```bash
git clone <repository-url>
cd zapier-calendar-agent
```

### Step 2: Install Dependencies
```bash
pip install openai fastmcp python-dotenv rich langchain
```

### Step 3: Set Up Environment Variables
Create a `.env` file in the project directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
ZAPIER_MCP_URL=your_zapier_mcp_server_url
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for AI intent analysis |
| `ZAPIER_MCP_URL` | Yes | Your Zapier MCP server endpoint URL |

### Zapier Setup Requirements

Your Zapier MCP server must have the following Google Calendar tools configured:

1. **google_calendar_find_event** - Find and list calendar events
2. **google_calendar_quick_add_event** - Create events from natural language
3. **google_calendar_create_detailed_event** - Create events with specific details
4. **google_calendar_find_busy_periods_in_calendar** - Check availability and conflicts

### Google Calendar Permissions

Ensure your Zapier integration has the following Google Calendar permissions:
- Read calendar events
- Create calendar events
- Access calendar metadata
- Read busy/free information

## Usage

### Starting the Agent

```bash
python zapier_calendar_agent.py
```

### Available Commands

#### Calendar Operations
| Command | Description | Example |
|---------|-------------|---------|
| Find Events | Search for calendar events | "What's on my calendar today?" |
| Create Events | Schedule new meetings | "Schedule a meeting with John at 2pm tomorrow" |
| Check Availability | Find busy periods | "Am I free this afternoon?" |

#### Diagnostic Commands
| Command | Description |
|---------|-------------|
| `test` | Run connection diagnostics and test calendar access |
| `tools` | List all available MCP tools |
| `memory` | Show conversation history and context |
| `debug [message]` | Test intent analysis without executing tools |
| `quit` or `exit` | End the session |

### Example Interactions

#### Finding Events
```
You: What's on my calendar today?

AI Agent:
I'll search your calendar for today's events.

Your Calendar Events:
- **Team Standup** at Jul 10, 2025 09:00AM (30m) [8 attendees]
- **Project Review** at Jul 10, 2025 02:00PM (1h)
- **1:1 with Manager** at Jul 10, 2025 04:00PM (30m)

Total time: 2h 0m
```

#### Creating Events
```
You: Schedule a meeting with Sarah tomorrow at 3pm for 1 hour

AI Agent:
I'll create a calendar event for you.

Event created successfully!
Title: Meeting with Sarah
Time: Jul 11, 2025 03:00PM
Event ID: abc123xyz789
View in Calendar: https://calendar.google.com/...
```

#### Checking Availability
```
You: Am I free between 2pm and 4pm today?

AI Agent:
I'll check your availability for that time period.

Your busy periods:
- Busy from Jul 10, 2025 02:00PM to Jul 10, 2025 03:00PM (Project Review)

You have 1 conflicting event
```

## Architecture

### System Components

#### AICalendarIntentAnalyzer
- **Purpose**: Analyzes user input and determines calendar actions
- **Technology**: OpenAI GPT-4o-mini for natural language understanding
- **Output**: Structured JSON with action type, tool selection, and parameters
- **Features**: Timezone awareness, date parsing, confidence scoring

#### ZapierAIAgent
- **Purpose**: Main orchestrator that manages the conversation flow
- **Responsibilities**: MCP communication, memory management, response formatting
- **Integration**: FastMCP client for Zapier communication
- **Memory**: LangChain ConversationBufferMemory for context retention

### Data Flow

1. **User Input**: Natural language calendar request
2. **Intent Analysis**: AI determines action type and parameters
3. **Tool Execution**: MCP call to appropriate Zapier calendar tool
4. **Response Processing**: Format results for user display
5. **Memory Storage**: Save interaction context for future reference

### Key Technologies

- **OpenAI API**: GPT-4o-mini for intent analysis and natural language processing
- **FastMCP**: Client library for Model Context Protocol communication
- **Rich**: Terminal formatting and beautiful console output
- **LangChain**: Conversation memory and message management
- **Zapier MCP**: Calendar tool integration and API access

## Troubleshooting

### Common Issues

#### Connection Problems
**Issue**: "MCP client not initialized"
**Solutions**:
- Verify `ZAPIER_MCP_URL` is set correctly in your `.env` file
- Check that your Zapier MCP server is running and accessible
- Test network connectivity to the Zapier endpoint

**Issue**: "Calendar access failed"
**Solutions**:
- Verify Google Calendar permissions in your Zapier account
- Check that calendar tools are properly configured in Zapier
- Ensure your Google account has calendar access enabled

#### API Issues
**Issue**: "OpenAI client not initialized"
**Solutions**:
- Verify `OPENAI_API_KEY` is set in your `.env` file
- Check that your OpenAI API key is valid and active
- Ensure you have sufficient API credits

**Issue**: "AI returned invalid JSON"
**Solutions**:
- Check OpenAI API status and availability
- Verify your API key has access to GPT-4o-mini
- Try the `debug` command to see raw AI responses

#### Calendar Data Issues
**Issue**: "No events found" when events exist
**Solutions**:
- Run `test` command to verify calendar connectivity
- Check timezone settings in both Google Calendar and the code
- Verify date range parameters in your query

### Diagnostic Commands

Use these commands to diagnose and fix issues:

```bash
# Test all connections and permissions
test

# See what tools are available
tools

# Debug intent analysis for a specific query
debug "What's on my calendar tomorrow?"

# View conversation history and context
memory
```

### Debug Output Interpretation

The agent provides detailed console output:
- **Intent Analysis**: Shows detected action and confidence level
- **Tool Execution**: Displays parameters sent to Zapier
- **Raw Results**: Shows truncated API responses
- **Error Messages**: Detailed error information with context

## Development

### Code Structure

#### Main Classes
```
AICalendarIntentAnalyzer
├── analyze_intent()     # Process user input and generate parameters
└── client              # OpenAI API client

ZapierAIAgent
├── __init__()          # Initialize components and connections
├── test_mcp_connection() # Diagnostic testing
├── connect_to_mcp()    # Establish MCP connection
├── execute_tool()      # Execute Zapier calendar tools
├── get_ai_response()   # Main conversation orchestration
└── run()               # Main application loop
```

#### Key Methods

**analyze_intent(user_input)**
- Processes natural language input
- Returns structured action and parameters
- Handles timezone conversion and date parsing

**execute_tool(tool_name, parameters)**
- Executes MCP tools via Zapier
- Handles response parsing and error recovery
- Provides simulated responses for development

**get_ai_response(user_message)**
- Orchestrates complete response generation
- Integrates intent analysis, tool execution, and formatting
- Manages conversation memory and context

### Adding New Features

#### New Calendar Operations
1. Add new action to `AVAILABLE_ACTIONS` in `AICalendarIntentAnalyzer`
2. Define tool parameters in `AVAILABLE_ZAPIER MCP TOOLS`
3. Add handling logic in the `get_ai_response()` method
4. Update examples and documentation

#### Modifying AI Behavior
Edit the system prompt in `analyze_intent()`:
- Change date parsing rules and timezone handling
- Add new tool parameters and response formats
- Modify confidence thresholds and reasoning logic
- Update examples for better intent recognition

#### Extending Memory System
- Implement persistent storage for conversation history
- Add user preference learning and personalization
- Create context-aware response generation
- Build cross-session memory capabilities

### Customization Options

#### Timezone Configuration
To change from Pacific Time (UTC-7):
```python
# In analyze_intent() method
pacific_tz = timezone(timedelta(hours=-5))  # Eastern Time (UTC-5)
current_context = f"Current date and time: {local_now.strftime('%Y-%m-%d %H:%M:%S %A')} (Eastern Time)"
```

#### Response Formatting
Modify the event formatting logic in `get_ai_response()`:
- Change date/time display formats
- Customize event information display
- Adjust summary statistics and totals

#### Simulated Responses
Update the fallback responses in `execute_tool()` for testing:
- Modify simulated event data
- Add new response scenarios
- Test edge cases and error conditions

## API Information

### Usage and Costs

#### OpenAI API
- **Model**: GPT-4o-mini
- **Cost**: Approximately $0.15 per 1M input tokens
- **Usage**: 1-3 API calls per user interaction
- **Optimization**: Low temperature (0.1) for consistent responses

#### Zapier API
- **Rate Limits**: Depends on your Zapier plan
- **Tool Calls**: 1 call per calendar operation
- **Monitoring**: Check Zapier dashboard for usage statistics

### Performance Characteristics
- **Intent Analysis**: 1-3 seconds per request
- **Tool Execution**: Varies based on calendar size and complexity
- **Memory Usage**: In-memory storage only, no persistence
- **Concurrency**: Single-threaded, processes one request at a time

## Security and Privacy

### Data Handling
- **API Keys**: Stored in environment variables, never in code
- **Calendar Data**: Processed in memory, not stored permanently
- **Conversation History**: Kept in memory during session only
- **User Input**: Sent to OpenAI for processing (see OpenAI privacy policy)

### Security Best Practices
- Store API keys securely in `.env` files
- Never commit credentials to version control
- Regularly rotate API keys and monitor usage
- Set up billing alerts and usage limits
- Review API access logs periodically

### Privacy Considerations
- Calendar data is accessed only for requested operations
- No data persistence beyond the current session
- OpenAI may process user queries according to their data policy
- Zapier handles calendar permissions according to Google's policies

## Dependencies

### Core Dependencies
```
openai>=1.0.0          # OpenAI API client for GPT-4o-mini
fastmcp>=0.1.0         # MCP protocol client
python-dotenv>=1.0.0   # Environment variable management
rich>=13.0.0           # Rich terminal formatting
langchain>=0.1.0       # Conversation memory management
```

### Standard Library
```
asyncio                # Asynchronous programming support
datetime               # Date and time handling with timezone support
json                   # JSON parsing and formatting
os                     # Environment variable access
```

### Installation Command
```bash
pip install openai fastmcp python-dotenv rich langchain
```

## Contributing

### Development Setup

1. **Fork the repository**
   ```bash
   git fork <repository-url>
   cd zapier-calendar-agent
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Set up pre-commit hooks** (if configured)
   ```bash
   pre-commit install
   ```

### Contribution Guidelines

#### Code Quality
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public methods
- Write clear, descriptive variable names

#### Testing
- Test new features with both real and simulated data
- Verify calendar operations don't create duplicate events
- Check error handling and edge cases
- Test with different timezone configurations

#### Documentation
- Update README for new features
- Add docstrings for new methods
- Include usage examples
- Document configuration changes

### Submitting Changes

1. Create a feature branch from main
2. Implement your changes with tests
3. Update documentation as needed
4. Submit a pull request with description
5. Respond to code review feedback

## License

MIT License - see LICENSE file for complete terms and conditions.

## Support and Resources

### Getting Help
1. **Check troubleshooting section** for common issues
2. **Run diagnostic commands** (`test`, `debug`, `tools`)
3. **Review console logs** for detailed error information
4. **Create GitHub issue** with reproduction steps and logs

### Useful Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Zapier Developer Platform](https://platform.zapier.com/)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [Rich Library Documentation](https://rich.readthedocs.io/)

### Community
- GitHub Issues for bug reports and feature requests
- Discussions for general questions and improvements
- Pull Requests for code contributions

---

**Version**: 1.0.0  
**Last Updated**: July 2025  
**Python Version**: 3.8+
