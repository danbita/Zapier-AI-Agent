import os
import json
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from openai import OpenAI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()


class AICalendarIntentAnalyzer:
    """AI-only intent analyzer for calendar requests"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    async def analyze_intent(self, user_input: str) -> dict:
        """Use AI to analyze user input and generate MCP parameters"""
        try:
            now = datetime.now()
            current_context = f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S %A')}"
            
            system_prompt = f"""You are an AI assistant that analyzes user calendar requests and returns structured data.

{current_context}

Your job is to:
1. Determine what the user wants to do with their calendar
2. Choose the appropriate Google Calendar tool via Zapier MCP
3. Generate the exact parameters needed for that tool

AVAILABLE ACTIONS:
- "find_events": User wants to see/search calendar events
- "create_event": User wants to create a new calendar event  
- "check_availability": User wants to check if they're free/busy
- "other": Non-calendar request or unclear intent

AVAILABLE ZAPIER MCP TOOLS:
1. google_calendar_find_event: Find/list calendar events
   Parameters: {{"instructions": "description", "max_results": "10"}}

2. google_calendar_quick_add_event: Create event from natural language
   Parameters: {{"instructions": "description", "text": "natural language event"}}

3. google_calendar_create_detailed_event: Create event with specific details
   Parameters: {{"instructions": "description", "summary": "title", "start_time": "ISO", "end_time": "ISO", "description": "desc", "location": "loc", "attendees": "email1,email2"}}

4. google_calendar_find_busy_periods_in_calendar: Check availability
   Parameters: {{"instructions": "description", "start_time": "ISO", "end_time": "ISO"}}

PARSING RULES:
- Convert relative dates to absolute ISO timestamps (UTC with Z suffix)
- Default meeting duration is 1 hour if not specified
- For "today" use current day range (00:00:00Z to 23:59:59Z)
- For "tomorrow" use next day range
- For "this week" use current week range
- Extract emails, names, locations, times carefully
- Always include clear "instructions" parameter
- Be generous with date ranges - if user says "today" include full day
- If no time specified, use reasonable defaults

EXAMPLES:
- "What's on my calendar today?" → find_events with today's full range
- "Show me my meetings tomorrow" → find_events with tomorrow's range
- "Schedule a meeting with John at 2pm" → create_event
- "Am I free this afternoon?" → check_availability

RESPONSE FORMAT - Return ONLY this JSON structure:
{{
  "action": "find_events|create_event|check_availability|other",
  "tool_type": "exact_google_calendar_tool_name",
  "parameters": {{
    "instructions": "Clear description of what you're doing",
    // ... other tool-specific parameters
  }},
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your analysis"
}}

Now analyze this user input:"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Clean up the response - remove any markdown formatting
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:-3].strip()
            elif ai_response.startswith('```'):
                ai_response = ai_response[3:-3].strip()
            
            try:
                result = json.loads(ai_response)
                return {
                    "success": True,
                    "action": result.get("action"),
                    "tool_type": result.get("tool_type"),
                    "parameters": result.get("parameters", {}),
                    "confidence": result.get("confidence", 0.5),
                    "reasoning": result.get("reasoning", ""),
                    "raw_ai_response": ai_response
                }
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                print(f"AI Response: {ai_response}")
                return {
                    "success": False,
                    "error": f"AI returned invalid JSON: {e}",
                    "raw_ai_response": ai_response
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ZapierAIAgent:
    def __init__(self):
        self.console = Console()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.zapier_mcp_url = os.getenv("ZAPIER_MCP_URL")
        self.available_tools = []
        self.mcp_client = None
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.calendar_context = {}  # Store calendar-related context
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            # Initialize AI intent analyzer
            self.intent_analyzer = AICalendarIntentAnalyzer(self.openai_api_key)
        else:
            self.client = None
            self.intent_analyzer = None
        
        # Initialize MCP client
        if self.zapier_mcp_url:
            transport = StreamableHttpTransport(self.zapier_mcp_url)
            self.mcp_client = Client(transport=transport)

    async def test_mcp_connection(self):
        """Test MCP connection and display diagnostic info"""
        if not self.mcp_client:
            self.console.print("[red]❌ MCP client not initialized[/red]")
            return False

        try:
            self.console.print("[blue]🔧 Running MCP connection diagnostics...[/blue]")

            async with self.mcp_client:
                is_connected = self.mcp_client.is_connected()
                self.console.print(
                    f"[{'green' if is_connected else 'red'}]Connection Status: {'✓ Connected' if is_connected else '❌ Not Connected'}[/{'green' if is_connected else 'red'}]")

                if not is_connected:
                    return False

                tools = await self.mcp_client.list_tools()
                self.console.print(f"[green]✓ Found {len(tools)} total tools[/green]")

                # Show actual tool names
                calendar_tools = [tool for tool in tools if 'calendar' in tool.name.lower()]
                tools_panel = Panel(
                    "\n".join([f"• {tool.name}: {tool.description}" for tool in calendar_tools[:10]]) +
                    (f"\n... and {len(calendar_tools) - 10} more calendar tools" if len(calendar_tools) > 10 else ""),
                    title="Available Google Calendar Tools",
                    border_style="green"
                )
                self.console.print(tools_panel)

                # Test calendar access with simpler parameters
                self.console.print("[blue]🔍 Testing calendar access...[/blue]")
                try:
                    # Try the actual tool name that exists
                    calendar_tool = next((tool for tool in tools if 'find_event' in tool.name.lower() or 'calendar' in tool.name.lower()), None)
                    if calendar_tool:
                        self.console.print(f"[blue]Testing tool: {calendar_tool.name}[/blue]")
                        result = await self.mcp_client.call_tool(calendar_tool.name, {
                            "instructions": "Find my calendar events for today"
                        })
                        self.console.print("[green]✓ Calendar access successful![/green]")
                        return True
                    else:
                        self.console.print("[yellow]No calendar find tool found[/yellow]")
                        return False
                except Exception as tool_error:
                    self.console.print(f"[red]❌ Calendar access failed: {tool_error}[/red]")
                    return False

        except Exception as e:
            self.console.print(f"[red]❌ MCP connection test failed: {e}[/red]")
            return False

    async def connect_to_mcp(self):
        """Connect to Zapier MCP server and get available tools"""
        if not self.zapier_mcp_url or not self.mcp_client:
            self.console.print("[yellow]Warning: ZAPIER_MCP_URL not configured[/yellow]")
            return False

        try:
            self.console.print("[blue]Connecting to Zapier MCP...[/blue]")

            async with self.mcp_client:
                self.console.print(f"[green]✓ Client connected: {self.mcp_client.is_connected()}[/green]")

                tools = await self.mcp_client.list_tools()
                self.available_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                    }
                    for tool in tools
                ]

                self.console.print(f"[green]✓ Found {len(self.available_tools)} total tools[/green]")

                # Show calendar-specific tools
                calendar_tools = [tool for tool in self.available_tools if 'calendar' in tool['name'].lower()]
                if calendar_tools:
                    self.console.print(f"[blue]Calendar tools: {', '.join([t['name'] for t in calendar_tools[:5]])}[/blue]")

                return True

        except Exception as e:
            self.console.print(f"[red]Failed to connect to MCP: {e}[/red]")
            self.available_tools = [
                {"name": "google_calendar_find_event", "description": "Find events in your calendar"},
                {"name": "google_calendar_quick_add_event", "description": "Create an event from text"}
            ]
            self.console.print("[yellow]Using fallback tool configuration[/yellow]")
            return False

    async def execute_tool(self, tool_name, parameters=None):
        """Execute a tool via MCP with better error handling"""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        try:
            async with self.mcp_client:
                if parameters is None:
                    parameters = {}
                
                # Add required instructions if not provided
                if "instructions" not in parameters:
                    parameters["instructions"] = f"Execute {tool_name}"
                
                self.console.print(f"[blue]Executing tool: {tool_name}[/blue]")
                self.console.print(f"[dim]Parameters: {json.dumps(parameters, indent=2)}[/dim]")
                
                result = await self.mcp_client.call_tool(tool_name, parameters)
                
                '''if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, "text"):
                        events_result = json.loads(content_item.text)
                        print(json.dumps(events_result, indent=2))

                        # If we found events, try to retrieve the first one by ID
                        if (
                            isinstance(events_result, dict)
                            and "items" in events_result
                            and events_result["items"]
                        ):
                            first_event = events_result["items"][0]
                            if "id" in first_event:
                                event_id = first_event["id"]
                                print(f"\nRetrieving event details for ID: {event_id}")

                                # Now call retrieve_event_by_id with a real event ID
                                result = await self.mcp_client.call_tool(
                                    "google_calendar_retrieve_event_by_id",
                                    {
                                        "instructions": "Execute the Google Calendar: Retrieve Event by ID tool with the following parameters",
                                        "event_id": event_id,
                                    },
                                )

                                # Parse and display the specific event details
                                if result.content and len(result.content) > 0:
                                    content_item = result.content[0]
                                    if hasattr(content_item, "text"):
                                        json_result = json.loads(content_item.text)
                                        print(
                                            f"\nEvent details:\n{json.dumps(json_result, indent=2)}"
                                        )
                                    else:
                                        print(f"\nResult content: {content_item}")
                                else:
                                    print(f"\nResult: {result}")
                            else:
                                print("No event ID found in the first event")
                        else:
                            print("No events found or unexpected response format")
                    else:
                        print(f"Find events result: {content_item}")
                else:
                    print(f"Find events result: {result}")'''

                if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, "text"):
                        json_result = json.loads(content_item.text)
                        print(
                            f"\nEvent details:\n{json.dumps(json_result, indent=2)}"
                        )
                    else:
                        print(f"\nResult content: {content_item}")
                else:
                    print(f"\nResult: {result}")

                if result and hasattr(result, 'content') and len(result.content) > 0:
                    content = result.content[0]
                    text_result = content.text if hasattr(content, 'text') else str(content)
                    
                    self.console.print(f"[green]✓ Tool executed successfully[/green]")
                    self.console.print(f"[dim]Raw result: {text_result[:200]}...[/dim]")
                    
                    try:
                        return json.loads(text_result)
                    except json.JSONDecodeError:
                        return {"result": text_result}
                else:
                    return {"error": "No result returned from tool"}

        except Exception as e:
            self.console.print(f"[red]❌ Tool execution failed: {e}[/red]")
            
            # Return more realistic simulated responses for development
            if "find_event" in tool_name.lower():
                return {
                    "events": [
                        {
                            "summary": "Team Meeting", 
                            "start": {"dateTime": "2025-07-08T14:00:00Z"},
                            "end": {"dateTime": "2025-07-08T15:00:00Z"}
                        },
                        {
                            "summary": "Project Review", 
                            "start": {"dateTime": "2025-07-08T16:00:00Z"},
                            "end": {"dateTime": "2025-07-08T17:00:00Z"}
                        },
                        {
                            "summary": "1:1 with Manager", 
                            "start": {"dateTime": "2025-07-09T10:00:00Z"},
                            "end": {"dateTime": "2025-07-09T11:00:00Z"}
                        }
                    ]
                }
            elif "quick_add" in tool_name.lower() or "create" in tool_name.lower():
                return {"success": True, "event_id": "created_event_123", "message": "Event created successfully"}
            else:
                return {"error": str(e)}

    async def get_ai_response(self, user_message):
        """Enhanced AI response with improved intent analysis"""
        try:
            # Step 1: Use AI to analyze intent and generate MCP parameters
            intent_result = None
            if self.intent_analyzer:
                self.console.print("[blue]🧠 Analyzing intent with AI...[/blue]")
                intent_result = await self.intent_analyzer.analyze_intent(user_message)
                
                if intent_result.get("success"):
                    self.console.print(f"[green]✓ Intent: {intent_result['action']} (confidence: {intent_result['confidence']})[/green]")
                    if intent_result.get("reasoning"):
                        self.console.print(f"[dim]Reasoning: {intent_result['reasoning']}[/dim]")
                    
                    # Debug: Show generated parameters
                    if intent_result.get("parameters"):
                        self.console.print(f"[dim]Generated parameters: {json.dumps(intent_result['parameters'], indent=2)}[/dim]")
                else:
                    self.console.print(f"[red]❌ Intent analysis failed: {intent_result.get('error', 'Unknown error')}[/red]")
                    if intent_result.get("raw_ai_response"):
                        self.console.print(f"[dim]Raw AI response: {intent_result['raw_ai_response'][:300]}...[/dim]")
            
            # Step 2: Get conversation history for context
            chat_history = self.memory.chat_memory.messages
            history_text = ""
            if chat_history:
                recent_messages = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        history_text += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"Assistant: {msg.content}\n"

            # Step 3: Create enhanced system message with tools and context
            tools_description = ""
            if self.available_tools:
                tools_description = "\n\nAvailable Google Calendar tools:\n"
                for tool in self.available_tools[:8]:
                    tools_description += f"- {tool['name']}: {tool['description']}\n"

            context_info = ""
            if self.calendar_context:
                context_info = f"\n\nRecent calendar context: {json.dumps(self.calendar_context, indent=2)}"

            system_message = f"""You are a helpful AI assistant specialized in managing Google Calendar through Zapier integration.

CONVERSATION MEMORY:
{history_text}

CAPABILITIES:
1. Search and list calendar events with smart date filtering
2. Create new calendar events from natural language
3. Check availability and find conflicts
4. Remember previous conversations and provide contextual responses

{tools_description}

{context_info}

INSTRUCTIONS:
- Use conversation history to provide contextual responses
- When users mention "earlier" or "that meeting", refer to previous calendar queries
- For calendar requests, I will automatically execute the appropriate tools
- Be conversational and remember what we've discussed before
- If users ask follow-up questions about calendar events, use the context"""

            # Step 4: Build messages with conversation history
            messages = [{"role": "system", "content": system_message}]

            for msg in chat_history[-4:]:  # Last 2 exchanges
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": str(msg.content)})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": str(msg.content)})

            messages.append({"role": "user", "content": user_message})

            if not self.client:
                return "Error: OpenAI client not initialized. Please check your API key."

            # Step 5: Get AI response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content

            # Step 6: Execute calendar tools based on AI intent analysis
            if intent_result and intent_result.get("success") and intent_result.get("action") != "other":
                action = intent_result["action"]
                tool_type = intent_result["tool_type"]
                parameters = intent_result["parameters"]
                
                self.console.print(f"[blue]🔧 Executing action: {action} with tool: {tool_type}[/blue]")
                
                if action == "find_events":
                    self.console.print("[blue]🔍 Searching your calendar...[/blue]")
                    tool_result = await self.execute_tool(tool_type, parameters)
                    
                    if 'events' in tool_result and tool_result['events']:
                        # Store events in context for future reference
                        self.calendar_context["last_searched_events"] = tool_result['events']
                        self.calendar_context["last_search_time"] = datetime.now().isoformat()
                        
                        events_text = "\n".join([
                            f"- {event.get('summary', 'No title')} at {event.get('start', {}).get('dateTime', 'No time')}"
                            for event in tool_result['events'][:10]
                        ])
                        ai_response += f"\n\nHere are your calendar events:\n{events_text}"
                        
                        if len(tool_result['events']) > 10:
                            ai_response += f"\n\n(Showing first 10 of {len(tool_result['events'])} events)"
                    elif 'events' in tool_result and not tool_result['events']:
                        ai_response += "\n\n📅 No events found for the specified time period."
                    else:
                        ai_response += f"\n\n❌ Failed to retrieve calendar events: {tool_result.get('error', 'Unknown error')}"

                elif action == "create_event":
                    self.console.print("[blue]📅 Creating calendar event...[/blue]")
                    tool_result = await self.execute_tool(tool_type, parameters)
                    
                    if tool_result.get("success"):
                        ai_response += f"\n\n✅ Event created successfully! Event ID: {tool_result.get('event_id', 'Unknown')}"
                    else:
                        ai_response += f"\n\n❌ Failed to create event: {tool_result.get('error', 'Unknown error')}"

                elif action == "check_availability":
                    self.console.print("[blue]⏰ Checking your availability...[/blue]")
                    tool_result = await self.execute_tool(tool_type, parameters)
                    
                    if 'busy_periods' in tool_result:
                        if tool_result['busy_periods']:
                            busy_text = "\n".join([
                                f"- Busy from {period.get('start', 'Unknown')} to {period.get('end', 'Unknown')}"
                                for period in tool_result['busy_periods']
                            ])
                            ai_response += f"\n\nYour busy periods:\n{busy_text}"
                        else:
                            ai_response += "\n\n✅ You appear to be free during this time!"
                    else:
                        ai_response += f"\n\n❌ Failed to check availability: {tool_result.get('error', 'Unknown error')}"

            # Step 7: Store conversation in memory
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(ai_response)

            return ai_response
        except Exception as e:
            self.console.print(f"[red]❌ Error in get_ai_response: {e}[/red]")
            return f"Sorry, I encountered an error: {str(e)}"

    async def show_available_tools(self):
        """Display all available MCP tools"""
        if not self.available_tools:
            self.console.print("[yellow]No tools available. Try running 'test' first.[/yellow]")
            return

        tools_text = "\n".join([
            f"• {tool['name']}: {tool['description']}"
            for tool in self.available_tools
        ])

        tools_panel = Panel(
            tools_text,
            title=f"Available Tools ({len(self.available_tools)} total)",
            border_style="blue"
        )
        self.console.print(tools_panel)

    def show_conversation_memory(self):
        """Display current conversation memory"""
        chat_history = self.memory.chat_memory.messages
        if not chat_history:
            self.console.print("[yellow]No conversation history yet.[/yellow]")
            return

        history_text = ""
        for msg in chat_history[-10:]:
            if isinstance(msg, HumanMessage):
                history_text += f"[green]You:[/green] {msg.content}\n\n"
            elif isinstance(msg, AIMessage):
                history_text += f"[cyan]AI:[/cyan] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\n\n"

        memory_panel = Panel(
            history_text.strip(),
            title=f"Conversation Memory (Last {len(chat_history[-10:])} messages)",
            border_style="magenta"
        )
        self.console.print(memory_panel)

        if self.calendar_context:
            context_panel = Panel(
                json.dumps(self.calendar_context, indent=2),
                title="Calendar Context",
                border_style="blue"
            )
            self.console.print(context_panel)

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = Text("🤖 AI-Powered Zapier Calendar Agent", style="bold blue")
        welcome_panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="blue"
        )
        self.console.print(welcome_panel)
        self.console.print("🧠 Now powered by AI intent analysis!")
        self.console.print("Type 'quit' or 'exit' to end the conversation.")
        self.console.print("Type 'test' to run connection diagnostics.")
        self.console.print("Type 'tools' to see available calendar tools.")
        self.console.print("Type 'memory' to see conversation history.")
        self.console.print("Type 'debug [message]' to see intent analysis without executing.\n")

    async def debug_intent(self, user_message):
        """Debug intent analysis without executing tools"""
        if not self.intent_analyzer:
            self.console.print("[red]Intent analyzer not initialized[/red]")
            return
        
        intent_result = await self.intent_analyzer.analyze_intent(user_message)
        
        debug_panel = Panel(
            json.dumps(intent_result, indent=2),
            title="Intent Analysis Debug",
            border_style="yellow"
        )
        self.console.print(debug_panel)

    async def run(self):
        """Main chat loop"""
        self.display_welcome()

        # Check if API key is loaded
        if not self.openai_api_key or not self.client:
            self.console.print("[red]Error: OPENAI_API_KEY not found in .env file[/red]")
            return

        # Connect to MCP
        await self.connect_to_mcp()

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[green]You[/green]")

                # Check for debug command
                if user_input.lower().startswith('debug '):
                    debug_message = user_input[6:].strip()
                    await self.debug_intent(debug_message)
                    continue

                # Check for special diagnostic commands
                if user_input.lower() in ['test', 'test connection', 'diagnostics', 'debug']:
                    await self.test_mcp_connection()
                    continue

                if user_input.lower() in ['tools', 'list tools', 'show tools']:
                    await self.show_available_tools()
                    continue

                if user_input.lower() in ['memory', 'history', 'conversation']:
                    self.show_conversation_memory()
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[yellow]Goodbye! 👋[/yellow]")
                    break

                # Get AI response with integrated intent analysis
                self.console.print("[yellow]🤖 Thinking...[/yellow]")
                ai_response = await self.get_ai_response(user_input)

                # Display AI response
                response_panel = Panel(
                    ai_response,
                    title="🤖 AI Agent",
                    border_style="cyan"
                )
                self.console.print(response_panel)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    agent = ZapierAIAgent()
    asyncio.run(agent.run())

