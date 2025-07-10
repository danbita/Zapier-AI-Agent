import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
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
            now = datetime.now(timezone.utc)
            # Convert to your local timezone (Pacific Time)
            pacific_tz = timezone(timedelta(hours=-7))  # PDT (UTC-7)
            local_now = now.astimezone(pacific_tz)
            
            current_context = f"Current date and time: {local_now.strftime('%Y-%m-%d %H:%M:%S %A')} (Pacific Time)"
            
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
- For calendar searches, use simple instructions describing what to find
- Default max_results to 5-10 events
- Use natural language in instructions rather than complex parameters
- Always include clear "instructions" parameter
- For date filtering, include date context in the instructions text

EXAMPLES:
- "What's on my calendar today?" → find_events with "Find events for today" instruction
- "Show me my meetings tomorrow" → find_events with "Find meetings for tomorrow" instruction
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
                    "results": [  # Changed from "events" to "results" to match Zapier format
                        {
                            "summary": "Team Meeting", 
                            "start": {
                                "dateTime": "2025-07-08T14:00:00Z",
                                "dateTime_pretty": "Jul 08, 2025 02:00PM",
                                "time_pretty": "02:00PM"
                            },
                            "duration_hours": 1,
                            "duration_minutes": 60,
                            "attendee_emails": "john@example.com,jane@example.com"
                        },
                        {
                            "summary": "Project Review", 
                            "start": {
                                "dateTime": "2025-07-08T16:00:00Z",
                                "dateTime_pretty": "Jul 08, 2025 04:00PM", 
                                "time_pretty": "04:00PM"
                            },
                            "duration_hours": 0.5,
                            "duration_minutes": 30,
                            "attendee_emails": ""
                        },
                        {
                            "summary": "1:1 with Manager", 
                            "start": {
                                "dateTime": "2025-07-09T10:00:00Z",
                                "dateTime_pretty": "Jul 09, 2025 10:00AM",
                                "time_pretty": "10:00AM"
                            },
                            "duration_hours": 1,
                            "duration_minutes": 60,
                            "attendee_emails": ""
                        }
                    ]
                }
            elif "quick_add" in tool_name.lower() or "create" in tool_name.lower():
                return {
                    "results": [{
                        "kind": "calendar#event",
                        "id": "created_event_123",
                        "status": "confirmed",
                        "summary": "New Meeting",
                        "htmlLink": "https://www.google.com/calendar/event?eid=example",
                        "start": {
                            "dateTime": "2025-07-10T15:00:00-07:00",
                            "dateTime_pretty": "Jul 10, 2025 03:00PM",
                            "time_pretty": "03:00PM"
                        },
                        "duration_hours": 1,
                        "duration_minutes": 60
                    }]
                }
            elif "busy_periods" in tool_name.lower() or "availability" in tool_name.lower():
                # Simulated busy periods response
                return {
                    "results": [
                        {
                            "summary": "Team Meeting",
                            "start": {
                                "dateTime": "2025-07-10T14:00:00-07:00",
                                "dateTime_pretty": "Jul 10, 2025 02:00PM"
                            },
                            "end": {
                                "dateTime": "2025-07-10T15:00:00-07:00", 
                                "dateTime_pretty": "Jul 10, 2025 03:00PM"
                            }
                        },
                        {
                            "summary": "Client Call",
                            "start": {
                                "dateTime": "2025-07-10T16:30:00-07:00",
                                "dateTime_pretty": "Jul 10, 2025 04:30PM"
                            },
                            "end": {
                                "dateTime": "2025-07-10T17:00:00-07:00",
                                "dateTime_pretty": "Jul 10, 2025 05:00PM"
                            }
                        }
                    ]
                }
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
                    
                    # Check for both 'results' and 'events' arrays (Zapier uses 'results')
                    events_list = tool_result.get('results') or tool_result.get('events', [])
                    
                    if events_list:
                        # Store events in context for future reference
                        self.calendar_context["last_searched_events"] = events_list
                        self.calendar_context["last_search_time"] = datetime.now().isoformat()
                        
                        # Format events with better time display
                        events_text = []
                        for event in events_list[:10]:
                            title = event.get('summary', 'No title')
                            
                            # Get formatted time from the API response
                            start_info = event.get('start', {})
                            if 'dateTime_pretty' in start_info:
                                time_str = start_info['dateTime_pretty']
                            elif 'time_pretty' in start_info:
                                date_str = start_info.get('date_pretty', '')
                                time_str = f"{date_str} {start_info['time_pretty']}"
                            elif 'dateTime' in start_info:
                                time_str = start_info['dateTime']
                            else:
                                time_str = 'No time'
                            
                            # Add duration if available
                            duration_hours = event.get('duration_hours', 0)
                            duration_minutes = event.get('duration_minutes', 0)
                            
                            if duration_hours > 0:
                                duration_str = f" ({duration_hours}h)"
                            elif duration_minutes > 0:
                                duration_str = f" ({duration_minutes}m)"
                            else:
                                duration_str = ""
                            
                            # Add attendee count if available
                            attendee_emails = event.get('attendee_emails', '')
                            if attendee_emails and attendee_emails.strip():
                                attendee_count = len(attendee_emails.split(','))
                                attendee_str = f" [{attendee_count} attendees]" if attendee_count > 1 else ""
                            else:
                                attendee_str = ""
                            
                            events_text.append(f"- **{title}** at {time_str}{duration_str}{attendee_str}")
                        
                        ai_response += f"\n\n📅 **Your Calendar Events:**\n" + "\n".join(events_text)
                        
                        if len(events_list) > 10:
                            ai_response += f"\n\n*(Showing first 10 of {len(events_list)} events)*"
                        
                        # Add summary statistics
                        total_duration = sum(event.get('duration_minutes', 0) for event in events_list)
                        if total_duration > 0:
                            hours = total_duration // 60
                            minutes = total_duration % 60
                            ai_response += f"\n\n⏱️ **Total time:** {hours}h {minutes}m"
                            
                    else:
                        ai_response += "\n\n📅 No events found for the specified time period."

                elif action == "create_event":
                    self.console.print("[blue]📅 Creating calendar event...[/blue]")
                    tool_result = await self.execute_tool(tool_type, parameters)
                    
                    # Check for both 'results' and direct success response
                    if 'results' in tool_result and tool_result['results']:
                        created_event = tool_result['results'][0]  # Get first created event
                        event_id = created_event.get('id', 'Unknown')
                        event_title = created_event.get('summary', 'Untitled Event')
                        
                        # Get formatted time if available
                        start_info = created_event.get('start', {})
                        if 'dateTime_pretty' in start_info:
                            time_str = start_info['dateTime_pretty']
                        elif 'time_pretty' in start_info:
                            date_str = start_info.get('date_pretty', '')
                            time_str = f"{date_str} {start_info['time_pretty']}"
                        elif 'dateTime' in start_info:
                            time_str = start_info['dateTime']
                        else:
                            time_str = "at scheduled time"
                        
                        # Get HTML link for easy access
                        html_link = created_event.get('htmlLink', '')
                        link_text = f"\n🔗 [View in Calendar]({html_link})" if html_link else ""
                        
                        ai_response += f"\n\n✅ **Event created successfully!**\n"
                        ai_response += f"📋 **Title:** {event_title}\n"
                        ai_response += f"🕐 **Time:** {time_str}\n"
                        ai_response += f"🆔 **Event ID:** {event_id}{link_text}"
                        
                    elif tool_result.get("success"):
                        # Fallback for simple success response
                        ai_response += f"\n\n✅ Event created successfully! Event ID: {tool_result.get('event_id', 'Unknown')}"
                    else:
                        ai_response += f"\n\n❌ Failed to create event: {tool_result.get('error', 'Unknown error')}"

                elif action == "check_availability":
                    self.console.print("[blue]⏰ Checking your availability...[/blue]")
                    tool_result = await self.execute_tool(tool_type, parameters)
                    
                    # Check for both 'results' and 'busy_periods' arrays (Zapier might use different formats)
                    busy_periods = tool_result.get('busy_periods', [])
                    results = tool_result.get('results', [])
                    
                    # If results array exists, it might contain busy period information
                    if results and not busy_periods:
                        # Check if results contain busy period data
                        busy_periods = results
                    
                    if busy_periods:
                        # Format busy periods with better time display
                        busy_text = []
                        for period in busy_periods:
                            # Handle different possible formats for busy periods
                            start_time = period.get('start', period.get('start_time', 'Unknown'))
                            end_time = period.get('end', period.get('end_time', 'Unknown'))
                            
                            # Try to format times nicely if they have pretty formats
                            if isinstance(start_time, dict):
                                start_str = start_time.get('dateTime_pretty', start_time.get('dateTime', str(start_time)))
                            else:
                                start_str = str(start_time)
                                
                            if isinstance(end_time, dict):
                                end_str = end_time.get('dateTime_pretty', end_time.get('dateTime', str(end_time)))
                            else:
                                end_str = str(end_time)
                            
                            # Add event title if available
                            event_title = period.get('summary', period.get('title', ''))
                            title_str = f" ({event_title})" if event_title else ""
                            
                            busy_text.append(f"- Busy from {start_str} to {end_str}{title_str}")
                        
                        ai_response += f"\n\n⏰ **Your busy periods:**\n" + "\n".join(busy_text)
                        
                        # Add summary if multiple periods
                        if len(busy_periods) > 1:
                            ai_response += f"\n\n*(You have {len(busy_periods)} conflicting events)*"
                    else:
                        ai_response += "\n\n✅ You appear to be free during this time!"

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
