import os
import json
import asyncio
import re
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


class ZapierAIAgent:
    def __init__(self):
        self.console = Console()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.zapier_mcp_url = os.getenv("ZAPIER_MCP_URL")
        self.available_tools = []
        self.mcp_client = None
        self.calendar_id = "dan.bita1@gmail.com"
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.calendar_context = {}  # Store calendar-related context
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None
        
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
                # Test 1: Check connection status
                is_connected = self.mcp_client.is_connected()
                self.console.print(f"[{'green' if is_connected else 'red'}]Connection Status: {'✓ Connected' if is_connected else '❌ Not Connected'}[/{'green' if is_connected else 'red'}]")
                if not is_connected:
                    return False
                # Test 2: List all available tools
                self.console.print("[blue]📋 Fetching available tools...[/blue]")
                tools = await self.mcp_client.list_tools()
                self.console.print(f"[green]✓ Found {len(tools)} total tools[/green]")
                # Display all tools with descriptions
                tools_panel = Panel(
                    "\n".join([f"• {tool.name}: {tool.description}" for tool in tools[:10]]) + \
                    (f"\n... and {len(tools) - 10} more tools" if len(tools) > 10 else ""),
                    title="Available Google Calendar Tools",
                    border_style="green"
                )
                self.console.print(tools_panel)
                # Test 3: Try a simple calendar operation
                self.console.print("[blue]🔍 Testing calendar access...[/blue]")
                try:
                    result = await self.mcp_client.call_tool("google_calendar_find_event", {})
                    self.console.print("[green]✓ Calendar access successful![/green]")
                    # Parse and display result - handle MCP result properly
                    if result and hasattr(result, 'content') and result.content:
                        text_result = result.content[0].text
                        try:
                            parsed_result = json.loads(text_result)
                            if 'events' in parsed_result or 'items' in parsed_result:
                                events = parsed_result.get('events', parsed_result.get('items', []))
                                self.console.print(f"[green]📅 Found {len(events)} calendar events[/green]")
                                # Show first few events
                                if events:
                                    sample_events = events[:3]
                                    events_text = "\n".join([
                                        f"• {event.get('summary', event.get('title', 'No title'))}"
                                        for event in sample_events
                                    ])
                                    sample_panel = Panel(
                                        events_text,
                                        title="Sample Events",
                                        border_style="cyan"
                                    )
                                    self.console.print(sample_panel)
                            else:
                                self.console.print("[yellow]⚠️ No events found or unexpected response format[/yellow]")
                                self.console.print(f"[dim]Response: {text_result[:200]}...[/dim]")
                        except json.JSONDecodeError:
                            self.console.print(f"[yellow]⚠️ Non-JSON response: {text_result[:100]}...[/yellow]")
                    return True
                except Exception as tool_error:
                    self.console.print(f"[red]❌ Calendar access failed: {tool_error}[/red]")
                    self.console.print("[yellow]This might mean Google Calendar isn't properly connected to your Zapier MCP[/yellow]")
                    return False
        except Exception as e:
            self.console.print(f"[red]❌ MCP connection test failed: {e}[/red]")
            return False

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


    async def execute_tool(self, tool_name, parameters=None):
        """Execute a tool via MCP"""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}
        
        try:
            async with self.mcp_client:
                # Show the parameters being passed for debugging
                self.console.print(f"[dim]Executing {tool_name} with parameters: {json.dumps(parameters or {}, indent=2)}[/dim]")
                
                # Execute the tool with proper MCP client
                result = await self.mcp_client.call_tool(tool_name, parameters)
                
                # Parse the result - handle MCP result properly
                if result and hasattr(result, 'content') and result.content:
                    # Get the first text content item
                    text_content = None
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            text_content = content_item.text
                            break
                    
                    if text_content:
                        try:
                            return json.loads(text_content)
                        except json.JSONDecodeError:
                            return {"result": text_content}
                    else:
                        return {"error": "No text content in result"}
                else:
                    return {"error": "No result returned from tool"}
                    
        except Exception as e:
            # Print the parameters and raise the error
            self.console.print(f"[red]Tool execution failed for {tool_name}[/red]")
            self.console.print(f"[red]Parameters were: {json.dumps(parameters or {}, indent=2)}[/red]")
            self.console.print(f"[red]Error: {str(e)}[/red]")
            return {"error": str(e)}

            
    def _parse_time_from_message(self, message):
        """Extract time information from user message"""
        
        message_lower = message.lower()
        time_info = {
            "has_time": False,
            "start_time": None,
            "end_time": None,
            "date_reference": None,
            "duration": None
        }
        
        # Date references - determine base date first
        if 'today' in message_lower:
            base_date = datetime.now()
            time_info["date_reference"] = "today"
        elif 'tomorrow' in message_lower:
            base_date = datetime.now() + timedelta(days=1)
            time_info["date_reference"] = "tomorrow"
        elif 'next week' in message_lower or 'this week' in message_lower:
            base_date = datetime.now()
            time_info["date_reference"] = "week"
        else:
            base_date = datetime.now() + timedelta(days=1)  # Default to tomorrow
            time_info["date_reference"] = "default"
        
        # Time pattern matching
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(am|pm)',  # 3:30 pm
            r'(\d{1,2})\s*(am|pm)',         # 3 pm
            r'at\s+(\d{1,2}):(\d{2})\s*(am|pm)',  # at 3:30 pm
            r'at\s+(\d{1,2})\s*(am|pm)',    # at 3 pm
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, message_lower)
            if match:
                time_info["has_time"] = True
                groups = match.groups()
                
                # Handle different group patterns
                if len(groups) == 3:  # Hour, minute, am/pm
                    hour = int(groups[0])
                    minute = int(groups[1])
                    period = groups[2].lower()
                elif len(groups) == 2:  # Hour, am/pm (no minutes)
                    hour = int(groups[0])
                    minute = 0
                    period = groups[1].lower()
                else:
                    continue  # Skip invalid patterns
                
                # Convert to 24-hour format
                if period == 'pm' and hour != 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                
                # Set the time on the base date
                final_datetime = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                time_info["start_time"] = final_datetime.isoformat() + "Z"
                
                # Default 1-hour duration
                end_time = final_datetime + timedelta(hours=1)
                time_info["end_time"] = end_time.isoformat() + "Z"
                break
        
        # Duration parsing
        duration_patterns = [
            r'for\s+(\d+)\s+hour[s]?',      # for 2 hours
            r'for\s+(\d+)\s+min[utes]*',    # for 30 minutes
            r'(\d+)\s+hour[s]?\s+meeting',  # 2 hour meeting
            r'(\d+)\s+min[utes]*\s+meeting' # 30 minute meeting
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, message_lower)
            if match and time_info["start_time"]:
                duration_value = int(match.group(1))
                
                if 'hour' in pattern:
                    duration_delta = timedelta(hours=duration_value)
                else:  # minutes
                    duration_delta = timedelta(minutes=duration_value)
                
                # Recalculate end time with custom duration
                start_dt = datetime.fromisoformat(time_info["start_time"].replace('Z', '+00:00'))
                end_dt = start_dt + duration_delta
                time_info["end_time"] = end_dt.isoformat().replace('+00:00', 'Z')
                time_info["duration"] = duration_value
                break
        
        return time_info
    
    def _extract_event_title(self, message):
        """Extract the actual event title from user message"""
        
        message_lower = message.lower()
        
        # Remove action words at the beginning
        title = message
        action_patterns = [
            r'^(create|add|schedule|book|make)\s+(an?\s+)?(event|meeting|appointment)\s+',
            r'^(could you|can you|please)\s+',
            r'^(create|add|schedule|book|make)\s+',
        ]
        
        for pattern in action_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Look for explicit naming patterns
        naming_patterns = [
            r'called\s+"([^"]+)"',           # called "Meeting Name"
            r"called\s+'([^']+)'",           # called 'Meeting Name'  
            r'called\s+([^,\s]+(?:\s+[^,\s]+)*?)(?:\s+(?:at|on|for|tomorrow|today)|\s*$)',  # called Meeting Name
            r'named\s+"([^"]+)"',            # named "Meeting Name"
            r"named\s+'([^']+)'",            # named 'Meeting Name'
            r'named\s+([^,\s]+(?:\s+[^,\s]+)*?)(?:\s+(?:at|on|for|tomorrow|today)|\s*$)',   # named Meeting Name
            r'for\s+"([^"]+)"',              # for "Meeting Name"
            r"for\s+'([^']+)'",              # for 'Meeting Name'
            r'"([^"]+)"\s+(?:at|on|for|tomorrow|today)',  # "Meeting Name" at 3pm
            r"'([^']+)'\s+(?:at|on|for|tomorrow|today)",  # 'Meeting Name' at 3pm
        ]
        
        for pattern in naming_patterns:
            match = re.search(pattern, title, flags=re.IGNORECASE)
            if match:
                extracted_title = match.group(1).strip()
                if len(extracted_title) > 1:
                    return extracted_title
        
        # If no explicit naming, try to extract the main subject
        # Remove time references
        title = re.sub(r'\bat\s+\d{1,2}(:\d{2})?\s*(am|pm)\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\b(today|tomorrow|next week|this week)\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\bfor\s+\d+\s+(hour|minute)s?\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\bon\s+\w+day\b', '', title, flags=re.IGNORECASE)
        
        # Remove common words that don't belong in titles
        title = re.sub(r'\b(with|and|the|a|an)\s+', ' ', title, flags=re.IGNORECASE)
        
        # Clean up
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Check if we have a meaningful title left
        if len(title) > 2 and not re.match(r'^(at|on|for|in)?\s*$', title, flags=re.IGNORECASE):
            return title
        
        # Default fallback
        return "Sample Event"

    def _create_natural_language_event(self, original_message, title=None):
        """Create a natural language event description for Google Calendar"""
        
        if title and title != "Sample Event":
            # We have a specific title, create natural language around it
            message_lower = original_message.lower()
            
            # Extract time information
            time_match = re.search(r'\bat\s+(\d{1,2}(:\d{2})?\s*(am|pm))', message_lower)
            date_match = re.search(r'\b(today|tomorrow|next week|this week)\b', message_lower)
            duration_match = re.search(r'\bfor\s+(\d+\s+(hour|minute)s?)\b', message_lower)
            
            # Build natural language event
            event_parts = [title]
            
            if date_match:
                event_parts.append(date_match.group(1))
            
            if time_match:
                event_parts.append(f"at {time_match.group(1)}")
            
            if duration_match:
                event_parts.append(f"for {duration_match.group(1)}")
            
            return " ".join(event_parts)
        
        else:
            # No specific title, clean up the original message
            cleaned = original_message
            
            # Remove action words
            cleaned = re.sub(r'^(create|add|schedule|book|make)\s+(an?\s+)?(event|meeting|appointment)\s+', 
                            'Sample Event ', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^(create|add|schedule|book|make)\s+', 
                            'Sample Event ', cleaned, flags=re.IGNORECASE)
            
            return cleaned
    
    async def analyze_calendar_intent(self, user_message):
        """Analyze user message to determine calendar intent and extract parameters"""
        intent_analysis = {
            "action": None,
            "parameters": {},
            "needs_calendar_access": False
        }
        
        message_lower = user_message.lower()
        
        # Detect calendar-related intents
        if any(keyword in message_lower for keyword in ['calendar', 'events', 'schedule', 'meeting', 'appointment', 'day', 'date', 'today', 'tomorrow', 'time']):
            intent_analysis["needs_calendar_access"] = True
            
            # Determine action type
            if any(action in message_lower for action in ['list', 'show', 'what', 'find', 'get', 'see', 'have']):
                intent_analysis["action"] = "find_events"
                
                # Handle pure date questions
                if any(date_q in message_lower for date_q in ['what day', 'what date', 'today is', 'current date']):
                    intent_analysis["action"] = "get_current_date"
                    intent_analysis["parameters"] = {
                        "instructions": "Get current date information"
                    }
                # Extract time parameters WITH instructions parameter
                elif 'today' in message_lower:
                    today = datetime.now()
                    intent_analysis["parameters"] = {
                        "instructions": f"Find my calendar events for today ({today.strftime('%Y-%m-%d')})",
                        "start_time": today.replace(hour=0, minute=0, second=0).isoformat() + "Z",
                        "end_time": today.replace(hour=23, minute=59, second=59).isoformat() + "Z",
                        "calendarid": self.calendar_id
                    }
                elif 'tomorrow' in message_lower:
                    tomorrow = datetime.now() + timedelta(days=1)
                    intent_analysis["parameters"] = {
                        "instructions": f"Find my calendar events for tomorrow ({tomorrow.strftime('%Y-%m-%d')})",
                        "start_time": tomorrow.replace(hour=0, minute=0, second=0).isoformat() + "Z",
                        "end_time": tomorrow.replace(hour=23, minute=59, second=59).isoformat() + "Z",
                        "calendarid": self.calendar_id
                    }
                elif 'week' in message_lower:
                    now = datetime.now()
                    week_later = now + timedelta(days=7)
                    intent_analysis["parameters"] = {
                        "instructions": "Find my calendar events for the next week",
                        "start_time": now.isoformat() + "Z",
                        "end_time": week_later.isoformat() + "Z",
                        "calendarid": self.calendar_id
                    }
                else:
                    # Default case - include instructions
                    intent_analysis["parameters"] = {
                        "instructions": "Find my calendar events",
                        "calendarid": self.calendar_id
                    }
                    
            elif any(action in message_lower for action in ['create', 'add', 'schedule', 'book', 'make']):
                intent_analysis["action"] = "create_event"
                
                # Enhanced event creation with intelligent parsing
                time_info = self._parse_time_from_message(user_message)
                
                if time_info["has_time"] and time_info["start_time"] and time_info["end_time"]:
                    # Use detailed event creation for specific times
                    intent_analysis["action"] = "create_detailed_event"
                    
                    # Intelligent title extraction
                    title = self._extract_event_title(user_message)
                    
                    intent_analysis["parameters"] = {
                        "instructions": f"Create detailed calendar event: {title} on {time_info['start_time']} for {time_info['duration']} hours",
                        "summary": title,
                        "start_time": time_info["start_time"],
                        "end_time": time_info["end_time"],
                        "all_day": "false",
                        "description": f"Event created via AI agent: {user_message}"
                    }
                else:
                    # Enhanced natural language parsing with title extraction
                    title = self._extract_event_title(user_message)
                    
                    # Create better natural language text for Google Calendar
                    if title == "Sample Event":
                        # No specific title found, use the cleaned message
                        event_text = self._create_natural_language_event(user_message)
                    else:
                        # Use the extracted title with time info
                        event_text = self._create_natural_language_event(user_message, title)
                    
                    intent_analysis["parameters"] = {
                        "instructions": f"Create a calendar event: {event_text}",
                        "text": event_text
                    }
                            
            elif any(action in message_lower for action in ['free', 'available', 'busy', 'conflicts']):
                intent_analysis["action"] = "check_availability"
                # Add instructions for availability checking
                intent_analysis["parameters"] = {
                    "instructions": "Check my availability and find busy periods"
                }
        
        return intent_analysis

    async def get_ai_response(self, user_message):
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

            # Build messages array with history
            messages = [{"role": "system", "content": system_message}]
            
            # Add recent conversation history
            for msg in chat_history[-4:]:  # Last 2 exchanges
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
            
            # Add current message
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content

            # Execute calendar tools based on intent analysis
            if intent["needs_calendar_access"]:
                if intent["action"] == "get_current_date":
                    # Handle date questions directly without MCP tools
                    current_date = datetime.now()
                    date_info = current_date.strftime("%A, %B %d, %Y")
                    ai_response += f"\n\n📅 Today is {date_info}"
                
                elif intent["action"] == "find_events":
                    self.console.print("[blue]🔍 Searching your calendar...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_find_event", intent["parameters"])
                    
                    # Debug: Show the raw tool result
                    self.console.print(f"[dim]Raw tool result: {json.dumps(tool_result, indent=2)}[/dim]")
                    
                    if 'events' in tool_result and tool_result['events']:
                        # Store events in context for future reference
                        self.calendar_context["last_searched_events"] = tool_result['events']
                        self.calendar_context["last_search_time"] = datetime.now().isoformat()
                        
                        events_text = "\n".join([
                            f"- {event.get('summary', 'No title')} at {event.get('start', {}).get('dateTime', 'No time')}"
                            for event in tool_result['events'][:10]  # Limit to 10 events
                        ])
                        ai_response += f"\n\nHere are your calendar events:\n{events_text}"
                        
                        if len(tool_result['events']) > 10:
                            ai_response += f"\n\n(Showing first 10 of {len(tool_result['events'])} events)"
                    elif 'events' in tool_result and not tool_result['events']:
                        ai_response += "\n\n📅 No events found for the specified time period."
                    elif 'error' in tool_result:
                        ai_response += f"\n\n❌ Error searching calendar: {tool_result['error']}"
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response format from calendar search: {tool_result}"
                
                elif intent["action"] == "create_event":
                    self.console.print("[blue]📅 Creating calendar event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_quick_add_event", intent["parameters"])
                    
                    if tool_result.get("success"):
                        ai_response += f"\n\n✅ Event created successfully! Event ID: {tool_result.get('event_id', 'Unknown')}"
                    else:
                        ai_response += f"\n\n❌ Failed to create event: {tool_result.get('error', 'Unknown error')}"
                
                elif intent["action"] == "create_detailed_event":
                    self.console.print("[blue]📝 Creating detailed calendar event...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_create_detailed_event", intent["parameters"])
                    
                    # Debug: Show the raw tool result
                    self.console.print(f"[dim]Raw tool result: {json.dumps(tool_result, indent=2)}[/dim]")
                    
                    if 'results' in tool_result:
                        # Handle Zapier MCP response format
                        results = tool_result['results']
                        if results and any(results):  # Check if any non-empty results
                            ai_response += f"\n\n✅ Detailed event created successfully!"
                            
                            # Try to extract event details from results
                            for result in results:
                                if isinstance(result, dict) and result:
                                    event_id = result.get('event_id', result.get('id', 'Unknown'))
                                    event_url = result.get('event_url', result.get('htmlLink', ''))
                                    if event_id != 'Unknown':
                                        ai_response += f"\n🆔 Event ID: {event_id}"
                                    if event_url:
                                        ai_response += f"\n🔗 Event URL: {event_url}"
                                    break
                        else:
                            ai_response += f"\n\n✅ Detailed event created (confirmation pending)"
                            
                    elif tool_result.get("success"):
                        ai_response += f"\n\n✅ Detailed event created successfully!"
                        event_id = tool_result.get('event_id', 'Unknown')
                        if event_id != 'Unknown':
                            ai_response += f"\n🆔 Event ID: {event_id}"
                        event_url = tool_result.get('event_url', '')
                        if event_url:
                            ai_response += f"\n🔗 Event URL: {event_url}"
                            
                    elif 'error' in tool_result:
                        ai_response += f"\n\n❌ Failed to create detailed event: {tool_result['error']}"
                        
                    else:
                        ai_response += f"\n\n⚠️ Unexpected response from detailed event creation: {tool_result}"

                elif intent["action"] == "check_availability":
                    self.console.print("[blue]⏰ Checking your availability...[/blue]")
                    tool_result = await self.execute_tool("google_calendar_find_busy_periods_in_calendar", intent["parameters"])
                    
                    if 'busy_periods' in tool_result:
                        if tool_result['busy_periods']:
                            busy_text = "\n".join([
                                f"- Busy from {period['start']} to {period['end']}"
                                for period in tool_result['busy_periods']
                            ])
                            ai_response += f"\n\nYour busy periods:\n{busy_text}"
                        else:
                            ai_response += "\n\n✅ You appear to be free during this time!"

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
            title=f"Available Tools ({len(self.available_tools)} total)",
            border_style="blue"
        )
        self.console.print(tools_panel)

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = Text("🤖 Zapier AI Agent", style="bold blue")
        welcome_panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="blue"
        )
        self.console.print(welcome_panel)
        self.console.print("Type 'quit' or 'exit' to end the conversation.")
        self.console.print("Type 'test' to run connection diagnostics.")
        self.console.print("Type 'tools' to see available calendar tools.")
        self.console.print("Type 'memory' to see conversation history.\n")

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
