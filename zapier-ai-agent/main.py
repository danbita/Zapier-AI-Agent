import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage


# Load environment variables
load_dotenv()


class ZapierAIAgent:
    def __init__(self):
        self.console = Console()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.zapier_mcp_url = os.getenv("ZAPIER_MCP_URL")
        self.available_tools = []
        self.mcp_client = None

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages = True,
            memory_key = "chat_history"
        )
        self.calendar_context = {}  # Store calendar-related context

        # Initialize OpenAI client
        if self.openai_api_key:
            self.client = OpenAI(api_key = self.openai_api_key)
        else:
            self.client = None

        # Initialize MCP client
        if self.zapier_mcp_url:
            transport = StreamableHttpTransport(self.zapier_mcp_url)
            self.mcp_client = Client(transport = transport)

    def _parse_mcp_content(self, result) -> str:
        """Parse MCP tool result content safely"""
        if not result or not hasattr(result, 'content') or not result.content:
            return "No result returned from tool"
        
        content = result.content[0]
        
        # Handle different content types safely
        if hasattr(content, 'text') and content.text is not None:
            return content.text
        elif hasattr(content, 'type'):
            # Handle other content types
            if content.type == 'text':
                return getattr(content, 'text', str(content))
            else:
                return f"Content type {content.type}: {str(content)}"
        else:
            return str(content)

    async def test_mcp_connection(self):
        """Test MCP connection and display diagnostic info"""
        if not self.mcp_client:
            self.console.print("[red]❌ MCP client not initialized[/red]")
            return False

        try:
            self.console.print("[blue]🔧 Running MCP connection diagnostics...[/blue]")

            async with self.mcp_client:
                # Test 1: Check connection status
                is_connected = self.mcp_client.is_connected()
                self.console.print(
                    f"[{'green' if is_connected else 'red'}]Connection Status: {'✓ Connected' if is_connected else '❌ Not Connected'}[/{'green' if is_connected else 'red'}]")

                if not is_connected:
                    return False

                # Test 2: List all available tools
                self.console.print("[blue]📋 Fetching available tools...[/blue]")
                tools = await self.mcp_client.list_tools()

                self.console.print(f"[green]✓ Found {len(tools)} total tools[/green]")

                # Display all tools with descriptions
                tools_panel = Panel(
                    "\n".join([f"• {tool.name}: {tool.description}" for tool in tools[:10]]) +
                    (f"\n... and {len(tools) - 10} more tools" if len(tools) > 10 else ""),
                    title = "Available Google Calendar Tools",
                    border_style = "green"
                )
                self.console.print(tools_panel)

                # Test 3: Try a simple calendar operation
                self.console.print("[blue]🔍 Testing calendar access...[/blue]")

                # Try to find events (this should work if calendar is connected)
                try:
                    result = await self.mcp_client.call_tool("google_calendar_find_event", {
                        "instructions": "Find my calendar events for today"
                    })
                    self.console.print("[green]✓ Calendar access successful![/green]")

                    # Parse and display result
                    text_result = self._parse_mcp_content(result)
                    
                    try:
                        parsed_result = json.loads(text_result)
                        if 'events' in parsed_result or 'items' in parsed_result:
                            events = parsed_result.get('events', parsed_result.get('items', []))
                            self.console.print(
                                f"[green]📅 Found {len(events)} calendar events[/green]")

                            # Show first few events
                            if events:
                                sample_events = events[:3]
                                events_text = "\n".join([
                                    f"• {event.get('summary', event.get('title', 'No title'))}"
                                    for event in sample_events
                                ])
                                sample_panel = Panel(
                                    events_text,
                                    title = "Sample Events",
                                    border_style = "cyan"
                                )
                                self.console.print(sample_panel)
                        else:
                            self.console.print(
                                "[yellow]⚠️ No events found or unexpected response format[/yellow]")
                            self.console.print(f"[dim]Response: {text_result[:200]}...[/dim]")
                    except json.JSONDecodeError:
                        self.console.print(
                            f"[yellow]⚠️ Non-JSON response: {text_result[:100]}...[/yellow]")

                    return True

                except Exception as tool_error:
                    self.console.print(f"[red]❌ Calendar access failed: {tool_error}[/red]")
                    self.console.print(
                        "[yellow]This might mean Google Calendar isn't properly connected to your Zapier MCP[/yellow]")
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

            # Connect using the proper MCP client
            async with self.mcp_client:
                self.console.print(f"[green]✓ Client connected: {self.mcp_client.is_connected()}[/green]")

                # Get available tools
                tools = await self.mcp_client.list_tools()
                self.available_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": getattr(tool.inputSchema, 'properties', {}) if hasattr(tool,
                                                                                             'inputSchema') else {}
                    }
                    for tool in tools
                ]

                self.console.print(
                    f"[green]✓ Found {len(self.available_tools)} Google Calendar tools[/green]")

                # Display some key tools
                key_tools = [tool['name'] for tool in self.available_tools[:5]]  # Show first 5
                if key_tools:
                    self.console.print(f"[blue]Key tools: {', '.join(key_tools)}[/blue]")

                return True

        except Exception as e:
            self.console.print(f"[red]Failed to connect to MCP: {e}[/red]")
            # Use fallback configuration for development
            self.available_tools = [
                {"name": "google_calendar_find_event", "description": "Find events in your calendar"},
                {"name": "google_calendar_quick_add_event", "description": "Create an event from text"}
            ]
            self.console.print("[yellow]Using fallback tool configuration[/yellow]")
            return False

    async def execute_generic_tool(self, tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute any available tool with generic parameter handling"""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        # Validate tool exists
        if not self.available_tools:
            return {"error": "No tools available. Please connect to MCP first."}
        
        tool_info = next((tool for tool in self.available_tools if tool['name'] == tool_name), None)
        if not tool_info:
            return {"error": f"Tool '{tool_name}' not found in available tools"}

        try:
            async with self.mcp_client:
                # Use provided parameters or empty dict
                if parameters is None:
                    parameters = {}
                
                # Execute the tool
                result = await self.mcp_client.call_tool(tool_name, parameters)

                # Parse the result
                text_result = self._parse_mcp_content(result)
                
                try:
                    return json.loads(text_result)
                except json.JSONDecodeError:
                    return {"result": text_result}

        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """Get a list of all available tools with their descriptions"""
        if not self.available_tools:
            return []
        
        return [
            {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": tool.get('parameters', {})
            }
            for tool in self.available_tools
        ]

    def show_conversation_memory(self):
        """Display current conversation memory"""
        chat_history = self.memory.chat_memory.messages
        if not chat_history:
            self.console.print("[yellow]No conversation history yet.[/yellow]")
            return

        history_text = ""
        for i, msg in enumerate(chat_history[-10:]):  # Show last 10 messages
            if isinstance(msg, HumanMessage):
                history_text += f"[green]You:[/green] {msg.content}\n\n"
            elif isinstance(msg, AIMessage):
                history_text += f"[cyan]AI:[/cyan] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\n\n"

        memory_panel = Panel(
            history_text.strip(),
            title = f"Conversation Memory (Last {len(chat_history[-10:])} messages)",
            border_style = "magenta"
        )
        self.console.print(memory_panel)

        # Show calendar context if available
        if self.calendar_context:
            context_panel = Panel(
                json.dumps(self.calendar_context, indent = 2),
                title = "Calendar Context",
                border_style = "blue"
            )
            self.console.print(context_panel)

    async def execute_tool(self, tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool via MCP"""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        # Validate tool exists
        if not self.available_tools:
            return {"error": "No tools available. Please connect to MCP first."}
        
        tool_exists = any(tool['name'] == tool_name for tool in self.available_tools)
        if not tool_exists:
            return {"error": f"Tool '{tool_name}' not found in available tools"}

        try:
            async with self.mcp_client:
                # Prepare parameters with defaults for calendar tools
                if parameters is None:
                    parameters = {}
                
                # Add required instructions for calendar tools if not provided
                tool_defaults = {
                    "google_calendar_find_event": "Find my calendar events",
                    "google_calendar_quick_add_event": "Create a calendar event",
                    "google_calendar_find_busy_periods_in_calendar": "Check my availability",
                    "google_calendar_retrieve_event_by_id": "Retrieve a specific event by ID",
                    "google_calendar_add_attendee_s_to_event": "Add attendees to an event",
                    "google_calendar_delete_event": "Delete an event",
                    "google_calendar_create_calendar": "Create a new calendar",
                    "google_calendar_create_detailed_event": "Create a detailed calendar event"
                }
                
                if tool_name in tool_defaults and "instructions" not in parameters:
                    parameters["instructions"] = tool_defaults[tool_name]
                
                # Execute the tool with proper MCP client
                result = await self.mcp_client.call_tool(tool_name, parameters)

                # Parse the result using the helper method
                text_result = self._parse_mcp_content(result)
                
                try:
                    return json.loads(text_result)
                except json.JSONDecodeError:
                    return {"result": text_result}

        except Exception as e:
            self.console.print(f"[yellow]Tool execution failed: {e}[/yellow]")
            
            # Provide more specific error handling
            error_msg = str(e)
            if "MCP error" in error_msg:
                return {"error": f"MCP connection error: {error_msg}"}
            elif "Invalid arguments" in error_msg:
                return {"error": f"Invalid tool parameters: {error_msg}"}
            elif "tool not found" in error_msg.lower():
                return {"error": f"Tool '{tool_name}' not available"}
            elif "timeout" in error_msg.lower():
                return {"error": f"Tool execution timed out: {error_msg}"}
            else:
                return {"error": f"Tool execution failed: {error_msg}"}

    async def analyze_calendar_intent(self, user_message: str) -> Dict[str, Any]:
        """Analyze user message to determine calendar intent and extract parameters"""
        intent_analysis = {
            "action": None,
            "parameters": {},
            "needs_calendar_access": False
        }

        message_lower = user_message.lower()

        # Detect calendar-related intents
        if any(keyword in message_lower for keyword in
               ['calendar', 'events', 'schedule', 'meeting', 'appointment', 'attendee', 'invite']):
            intent_analysis["needs_calendar_access"] = True

            # Determine action type
            if any(action in message_lower for action in
                   ['list', 'show', 'what', 'find', 'get', 'see']):
                intent_analysis["action"] = "find_events"

                # Extract time parameters
                if 'today' in message_lower:
                    today = datetime.now()
                    intent_analysis["parameters"] = {
                        "instructions": f"Find my calendar events for today ({today.strftime('%Y-%m-%d')})",
                        "start_time": today.replace(hour = 0, minute = 0, second = 0).isoformat() + "Z",
                        "end_time": today.replace(hour = 23, minute = 59, second = 59).isoformat() + "Z"
                    }
                elif 'tomorrow' in message_lower:
                    tomorrow = datetime.now() + timedelta(days = 1)
                    intent_analysis["parameters"] = {
                        "instructions": f"Find my calendar events for tomorrow ({tomorrow.strftime('%Y-%m-%d')})",
                        "start_time": tomorrow.replace(hour = 0, minute = 0,
                                                       second = 0).isoformat() + "Z",
                        "end_time": tomorrow.replace(hour = 23, minute = 59,
                                                     second = 59).isoformat() + "Z"
                    }
                elif 'week' in message_lower:
                    now = datetime.now()
                    week_later = now + timedelta(days = 7)
                    intent_analysis["parameters"] = {
                        "instructions": f"Find my calendar events for the next week",
                        "start_time": now.isoformat() + "Z",
                        "end_time": week_later.isoformat() + "Z"
                    }
                else:
                    # Default case
                    intent_analysis["parameters"] = {
                        "instructions": "Find my calendar events"
                    }

            elif any(action in message_lower for action in
                     ['create', 'add', 'schedule', 'book', 'make']):
                intent_analysis["action"] = "create_event"
                # Extract event details from message
                intent_analysis["parameters"] = {
                    "instructions": user_message
                }

            elif any(action in message_lower for action in ['free', 'available', 'busy', 'conflicts']):
                intent_analysis["action"] = "check_availability"

            elif any(action in message_lower for action in ['retrieve', 'get by id', 'find by id', 'specific']):
                intent_analysis["action"] = "retrieve_event_by_id"
                # Extract event ID from message
                intent_analysis["parameters"] = {
                    "instructions": user_message
                }

            elif any(action in message_lower for action in ['invite', 'add attendee', 'add attendees', 'send invite']):
                intent_analysis["action"] = "add_attendees"
                # Extract attendee information from message
                intent_analysis["parameters"] = {
                    "instructions": user_message
                }

            elif any(action in message_lower for action in ['delete', 'remove', 'cancel']):
                intent_analysis["action"] = "delete_event"
                # Extract event information for deletion
                intent_analysis["parameters"] = {
                    "instructions": user_message
                }

            elif any(action in message_lower for action in ['create calendar', 'new calendar', 'make calendar']):
                intent_analysis["action"] = "create_calendar"
                # Extract calendar details from message
                intent_analysis["parameters"] = {
                    "instructions": user_message
                }

            elif any(action in message_lower for action in ['detailed', 'specific event', 'with details']):
                intent_analysis["action"] = "create_detailed_event"
                # Extract detailed event information
                intent_analysis["parameters"] = {
                    "instructions": user_message
                }

        return intent_analysis

    async def get_ai_response(self, user_message: str) -> str:
        """Enhanced AI response with conversation memory and better calendar integration"""
        try:
            # Analyze calendar intent
            intent = await self.analyze_calendar_intent(user_message)

            # Get conversation history
            chat_history = self.memory.chat_memory.messages
            history_text = ""
            if chat_history:
                recent_messages = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        history_text += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"Assistant: {msg.content}\n"

            # Create tools description for OpenAI
            tools_description = ""
            if self.available_tools:
                tools_description = "\n\nAvailable Google Calendar tools:\n"
                for tool in self.available_tools[:8]:  # Show first 8 tools
                    tools_description += f"- {tool['name']}: {tool['description']}\n"

            # Add calendar context if available
            context_info = ""
            if self.calendar_context:
                context_info = f"\n\nRecent calendar context: {json.dumps(self.calendar_context, indent = 2)}"

            system_message = f"""You are a helpful AI assistant specialized in managing Google Calendar through Zapier integration.

CONVERSATION MEMORY:
{history_text}

CAPABILITIES:
1. Search and list calendar events with smart date filtering
2. Create new calendar events from natural language (quick add)
3. Create detailed calendar events with specific fields
4. Retrieve specific events by ID
5. Add attendees to existing events
6. Delete events
7. Create new calendars
8. Check availability and find conflicts
9. Remember previous conversations and provide contextual responses

{tools_description}

{context_info}

INSTRUCTIONS:
- Use conversation history to provide contextual responses
- When users mention "earlier" or "that meeting", refer to previous calendar queries
- For calendar requests, I will automatically execute the appropriate tools
- Be conversational and remember what we've discussed before
- If users ask follow-up questions about calendar events, use the context
- Support all calendar operations: create, read, update, delete, and manage attendees"""

            # Build messages array with proper typing
            messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_message}]

            # Add recent conversation history
            for msg in chat_history[-4:]:  # Last 2 exchanges
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": str(msg.content)})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": str(msg.content)})

            # Add current message
            messages.append({"role": "user", "content": user_message})

            if not self.client:
                return "Error: OpenAI client not initialized. Please check your API key."

            response = self.client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = messages,
                max_tokens = 600,
                temperature = 0.7
            )

            ai_response = response.choices[0].message.content or "I'm sorry, I couldn't generate a response."

            # Execute calendar tools based on intent analysis
            if intent["needs_calendar_access"]:
                if intent["action"] == "find_events":
                    self.console.print("[blue]🔍 Searching your calendar...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_find_event",
                                                          intent["parameters"])

                    if 'events' in tool_result:
                        # Store events in context for future reference
                        events = tool_result['events']
                        
                        # Robust event handling
                        if isinstance(events, list) and events:
                            self.calendar_context["last_searched_events"] = events
                            self.calendar_context["last_search_time"] = datetime.now().isoformat()

                            # Safely process events with error handling
                            event_display_list = []
                            for i, event in enumerate(events[:10]):  # Limit to 10 events
                                try:
                                    if isinstance(event, dict):
                                        summary = event.get('summary', event.get('title', 'No title'))
                                        start_info = event.get('start', {})
                                        
                                        if isinstance(start_info, dict):
                                            date_time = start_info.get('dateTime', start_info.get('date', 'No time'))
                                        else:
                                            date_time = str(start_info) if start_info else 'No time'
                                        
                                        event_display_list.append(f"- {summary} at {date_time}")
                                    else:
                                        # Handle non-dict events
                                        event_display_list.append(f"- Event {i+1}: {str(event)[:50]}...")
                                except Exception as event_error:
                                    self.console.print(f"[yellow]Warning: Could not process event {i+1}: {event_error}[/yellow]")
                                    event_display_list.append(f"- Event {i+1}: [Error processing event]")

                            if event_display_list:
                                events_text = "\n".join(event_display_list)
                                ai_response += f"\n\nHere are your calendar events:\n{events_text}"

                                if len(events) > 10:
                                    ai_response += f"\n\n(Showing first 10 of {len(events)} events)"
                            else:
                                ai_response += "\n\nNo events found for the specified time period."
                                
                        elif isinstance(events, list) and not events:
                            ai_response += "\n\nNo events found for the specified time period."
                        else:
                            ai_response += f"\n\nUnexpected events format: {type(events)}"
                            
                    elif 'error' in tool_result:
                        ai_response += f"\n\n❌ Error retrieving events: {tool_result['error']}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response format from calendar tool"
                        self.console.print(f"[yellow]Debug: tool_result = {tool_result}[/yellow]")

                elif intent["action"] == "create_event":
                    self.console.print("[blue]📅 Creating calendar event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_quick_add_event",
                                                          intent["parameters"])

                    if isinstance(tool_result, dict):
                        if tool_result.get("success"):
                            event_id = tool_result.get('event_id', 'Unknown')
                            ai_response += f"\n\n✅ Event created successfully! Event ID: {event_id}"
                        elif 'error' in tool_result:
                            ai_response += f"\n\n❌ Failed to create event: {tool_result['error']}"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected response from event creation: {tool_result}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response type from event creation: {type(tool_result)}"

                elif intent["action"] == "check_availability":
                    self.console.print("[blue]⏰ Checking your availability...[/blue]")
                    tool_result = await self.execute_tool(
                        "google_calendar_find_busy_periods_in_calendar", intent["parameters"])

                    if isinstance(tool_result, dict) and 'busy_periods' in tool_result:
                        busy_periods = tool_result['busy_periods']
                        
                        if isinstance(busy_periods, list) and busy_periods:
                            # Safely process busy periods
                            busy_display_list = []
                            for i, period in enumerate(busy_periods):
                                try:
                                    if isinstance(period, dict):
                                        start_time = period.get('start', 'Unknown')
                                        end_time = period.get('end', 'Unknown')
                                        busy_display_list.append(f"- Busy from {start_time} to {end_time}")
                                    else:
                                        busy_display_list.append(f"- Busy period {i+1}: {str(period)[:50]}...")
                                except Exception as period_error:
                                    self.console.print(f"[yellow]Warning: Could not process busy period {i+1}: {period_error}[/yellow]")
                                    busy_display_list.append(f"- Busy period {i+1}: [Error processing period]")
                            
                            if busy_display_list:
                                busy_text = "\n".join(busy_display_list)
                                ai_response += f"\n\nYour busy periods:\n{busy_text}"
                            else:
                                ai_response += "\n\n✅ You appear to be free during this time!"
                        elif isinstance(busy_periods, list) and not busy_periods:
                            ai_response += "\n\n✅ You appear to be free during this time!"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected busy periods format: {type(busy_periods)}"
                    elif isinstance(tool_result, dict) and 'error' in tool_result:
                        ai_response += f"\n\n❌ Error checking availability: {tool_result['error']}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response from availability check: {tool_result}"

                elif intent["action"] == "retrieve_event_by_id":
                    self.console.print("[blue]🔍 Retrieving specific event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_retrieve_event_by_id",
                                                          intent["parameters"])

                    if isinstance(tool_result, dict):
                        if 'event' in tool_result:
                            event = tool_result['event']
                            if isinstance(event, dict):
                                summary = event.get('summary', 'No title')
                                start_info = event.get('start', {})
                                if isinstance(start_info, dict):
                                    date_time = start_info.get('dateTime', start_info.get('date', 'No time'))
                                else:
                                    date_time = str(start_info) if start_info else 'No time'
                                
                                ai_response += f"\n\n✅ Found event: {summary} at {date_time}"
                            else:
                                ai_response += f"\n\n✅ Event retrieved: {str(event)[:100]}..."
                        elif 'error' in tool_result:
                            ai_response += f"\n\n❌ Error retrieving event: {tool_result['error']}"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected response from event retrieval: {tool_result}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response type from event retrieval: {type(tool_result)}"

                elif intent["action"] == "add_attendees":
                    self.console.print("[blue]👥 Adding attendees to event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_add_attendee_s_to_event",
                                                          intent["parameters"])

                    if isinstance(tool_result, dict):
                        if tool_result.get("success"):
                            ai_response += f"\n\n✅ Attendees added successfully!"
                        elif 'error' in tool_result:
                            ai_response += f"\n\n❌ Failed to add attendees: {tool_result['error']}"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected response from adding attendees: {tool_result}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response type from adding attendees: {type(tool_result)}"

                elif intent["action"] == "delete_event":
                    self.console.print("[blue]🗑️ Deleting event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_delete_event",
                                                          intent["parameters"])

                    if isinstance(tool_result, dict):
                        if tool_result.get("success"):
                            ai_response += f"\n\n✅ Event deleted successfully!"
                        elif 'error' in tool_result:
                            ai_response += f"\n\n❌ Failed to delete event: {tool_result['error']}"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected response from event deletion: {tool_result}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response type from event deletion: {type(tool_result)}"

                elif intent["action"] == "create_calendar":
                    self.console.print("[blue]📅 Creating new calendar...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_create_calendar",
                                                          intent["parameters"])

                    if isinstance(tool_result, dict):
                        if tool_result.get("success"):
                            calendar_id = tool_result.get('calendar_id', 'Unknown')
                            ai_response += f"\n\n✅ Calendar created successfully! Calendar ID: {calendar_id}"
                        elif 'error' in tool_result:
                            ai_response += f"\n\n❌ Failed to create calendar: {tool_result['error']}"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected response from calendar creation: {tool_result}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response type from calendar creation: {type(tool_result)}"

                elif intent["action"] == "create_detailed_event":
                    self.console.print("[blue]📝 Creating detailed event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_create_detailed_event",
                                                          intent["parameters"])

                    if isinstance(tool_result, dict):
                        if tool_result.get("success"):
                            event_id = tool_result.get('event_id', 'Unknown')
                            ai_response += f"\n\n✅ Detailed event created successfully! Event ID: {event_id}"
                        elif 'error' in tool_result:
                            ai_response += f"\n\n❌ Failed to create detailed event: {tool_result['error']}"
                        else:
                            ai_response += f"\n\n⚠️ Unexpected response from detailed event creation: {tool_result}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response type from detailed event creation: {type(tool_result)}"

            # Store conversation in memory
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(ai_response)

            return ai_response
        except Exception as e:
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
            title = f"Available Tools ({len(self.available_tools)} total)",
            border_style = "blue"
        )
        self.console.print(tools_panel)

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = Text("🤖 Zapier AI Agent", style = "bold blue")
        welcome_panel = Panel(
            welcome_text,
            title = "Welcome",
            border_style = "blue"
        )
        self.console.print(welcome_panel)
        self.console.print("Type 'quit' or 'exit' to end the conversation.")
        self.console.print("Type 'test' to run connection diagnostics.")
        self.console.print("Type 'tools' to see available calendar tools.")
        self.console.print("Type 'memory' to see conversation history.\n")

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

                # Get AI response
                self.console.print("[yellow]🤖 Thinking...[/yellow]")
                ai_response = await self.get_ai_response(user_input)

                # Display AI response
                response_panel = Panel(
                    ai_response,
                    title = "🤖 AI Agent",
                    border_style = "cyan"
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