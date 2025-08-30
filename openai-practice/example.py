from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import json

# Ensure we load the .env that sits next to this file (not just repo root)
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

api_key=os.getenv('OPENAI_API_KEY')

# NEW: MCP server config from environment (no hardcoded secrets)
mcp_label = os.getenv('MCP_RUBE_LABEL', 'rube')
mcp_url = os.getenv('MCP_RUBE_URL')
mcp_auth_header = os.getenv('MCP_RUBE_AUTH_HEADER', 'Authorization')
mcp_auth_value = os.getenv('MCP_RUBE_AUTH_VALUE')
# Allow all tools by default: only set allow-list if explicitly provided
_mcp_tools_raw = os.getenv('MCP_RUBE_ALLOWED_TOOLS', '').strip()
mcp_allowed_tools = [s.strip() for s in _mcp_tools_raw.split(',') if s.strip()]
mcp_require_approval = os.getenv('MCP_REQUIRE_APPROVAL', 'never')

if not mcp_url:
  raise RuntimeError("MCP_RUBE_URL is not set. Provide your actual MCP server URL in the environment.")

client = OpenAI()

response = client.responses.create(
  model="gpt-5",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "# Role and Objective\nServe as a sophisticated digital assistant with integrated access to over 500 applications via the Rube MCP, efficiently assisting users by leveraging rube-enabled apps to fulfill their requests. Use only information retrieved from rube-connected apps or well-validated internal reasoning; do not fabricate information.\n\n# Instructions\n- Begin by interpreting the user's request, clarifying ambiguities if needed.\n- Start every response with a concise conceptual checklist (3–7 high-level bullets) outlining the main sub-tasks to be performed.\n- Clearly present your reasoning and the planned steps, specifying which rube-connected app(s) and tools are selected for the user's need, before initiating any actions.\n- Before any significant tool call or external action, announce the purpose and specify the minimal required inputs.\n- Use only tools and applications available via Rube MCP. If a required action cannot be performed, state the limitation and suggest alternatives when possible.\n- After each tool call or code edit, validate and summarize the result in 1–2 lines. If successful, proceed; if not, self-correct as needed, persisting through multi-step tasks until all aspects of the user’s objective are satisfied.\n- Conclude each interaction by summarizing the completed actions and presenting the outcome or final answer clearly to the user.\n\n# Output Format\nFollow this structure for every response:\n1. **Checklist:** Concise high-level checklist of sub-tasks.\n2. **Reasoning:** Briefly summarize your thought process and choice of tools/apps.\n3. **Actions/Results:** For each app used, list the actions performed and the results obtained, each with a short validation.\n4. **Conclusion:** Clearly state the final outcome or resolution to the user's request.\n- Adjust level of detail in Checklist and Reasoning based on task complexity; document all tools/apps and outcomes in Actions/Results.\n- Use concise, precise language throughout.\n\n# Example\nUser input: \"Book me a flight to New York next Monday morning.\"\n\nOutput:\n**Checklist:**\n- Interpret the flight booking request\n- Confirm destination and date\n- Select appropriate travel booking app\n- Search for flights\n- Book the most suitable flight\n\n**Reasoning:** Using 'TravelBookingApp' via Rube MCP to fulfill the user's request for a morning flight to New York next Monday, confirming details as needed.\n\n**Actions/Results:**\n- Queried TravelBookingApp for next Monday morning flights to New York. Validation: Flights retrieved successfully.\n- Booked the best matching flight. Validation: Booking confirmed, confirmation details received.\n\n**Conclusion:** Your flight to New York next Monday morning has been booked. You'll receive all details shortly.\n\n---\n# Important Reminders\n- Always plan and present your reasoning before acting.\n- Provide a high-level checklist before execution.\n- Announce the purpose and minimal inputs for any significant tool call before executing, and validate the results after each action.\n- Remain engaged and persist until the user's request is fully satisfied.\n- Set reasoning_effort = medium by default; adjust up for complex or ambiguous tasks. Make internal reasoning concise unless detailed analysis is necessary."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "get my latest emails from my gmail account"
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "text"
    },
    "verbosity": "medium"
  },
  reasoning={
    "effort": "high",
    "summary": "auto"
  },
  tools=[
    {
      "type": "web_search",
      "search_context_size": "medium",
    },
    (
      (lambda: (
        (lambda d: (mcp_allowed_tools and d.update({"allowed_tools": mcp_allowed_tools}) or d))({
          "type": "mcp",
          "server_label": mcp_label,
          "server_url": mcp_url,
          "headers": ({mcp_auth_header: mcp_auth_value} if mcp_auth_value else {}),
          "require_approval": mcp_require_approval,
        })
      ))()
    )
  ],
  store=True,
  include=[
    "reasoning.encrypted_content",
    "web_search_call.action.sources"
  ]
)

# Try to extract a compact Gmail summary from the response and print it.
# If extraction fails, fall back to a safe pretty-print of the whole response.
try:
  # Prefer SDK-provided dict/serialization helpers if available
  if hasattr(response, "to_dict"):
    serializable = response.to_dict()
  elif hasattr(response, "dict"):
    serializable = response.dict()
  else:
    # Best-effort: inspect common attributes for human-friendly output
    serializable = {}
    if hasattr(response, "output"):
      serializable["output"] = response.output
    if hasattr(response, "output_text"):
      serializable["output_text"] = response.output_text
    if hasattr(response, "text"):
      serializable["text"] = response.text
    # fallback to the full __dict__ when nothing else is available
    if not serializable:
      try:
        serializable = response.__dict__
      except Exception:
        serializable = str(response)

  # Recursive helpers to find message lists and keys
  def find_messages(obj):
    results = []
    if isinstance(obj, dict):
      for k, v in obj.items():
        if k == "messages" and isinstance(v, list) and v and isinstance(v[0], dict):
          results.append(v)
        else:
          results.extend(find_messages(v))
    elif isinstance(obj, list):
      for item in obj:
        results.extend(find_messages(item))
    return results

  def find_key(obj, key):
    if isinstance(obj, dict):
      if key in obj:
        return obj[key]
      for v in obj.values():
        found = find_key(v, key)
        if found is not None:
          return found
    elif isinstance(obj, list):
      for item in obj:
        found = find_key(item, key)
        if found is not None:
          return found
    return None

  msg_lists = find_messages(serializable)
  messages = msg_lists[0] if msg_lists else []

  if messages:
    summaries = []
    for m in messages:
      ts = m.get("messageTimestamp") or m.get("internalDate") or m.get("timestamp")
      sender = m.get("sender") or m.get("from") or m.get("fromAddress")
      subject = m.get("subject")
      # preview may be nested
      preview = ""
      pv = m.get("preview")
      if isinstance(pv, dict):
        preview = pv.get("body") or pv.get("snippet") or ""
      else:
        preview = pv or m.get("snippet") or m.get("messageText") or ""

      summaries.append({
        "messageId": m.get("messageId") or m.get("id"),
        "threadId": m.get("threadId"),
        "timestamp": ts,
        "sender": sender,
        "subject": subject,
        "preview": preview,
      })

    next_token = find_key(serializable, "nextPageToken") or find_key(serializable, "next_page_token")
    out = {"messages": summaries}
    if next_token:
      out["nextPageToken"] = next_token

    print(json.dumps(out, default=lambda o: str(o), indent=2))
  else:
    # No messages found; fall back to full pretty-print
    print(json.dumps(serializable, default=lambda o: str(o), indent=2))
except Exception as e:
  print("[Warning] Could not extract messages; falling back to pretty-print:", e)
  try:
    print(json.dumps(serializable, default=lambda o: str(o), indent=2))
  except Exception:
    print(str(response))