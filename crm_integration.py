"""
Voicera → GridCRM Integration Module

Writes call data directly to the GridCRM Firestore database when a Voicera call ends.
Uses the same Firebase project (grid-a4798) and the same Firestore schema that GridCRM uses,
so entries created here show up natively in the GridCRM mobile app.

Firestore collections used:
  - customers: { ownerId, name, phone, address, created_at }
  - calls:     { ownerId, customer_id, customer: {name, phone}, call_type, problem_description,
                 priority, status, technician_assigned, raw_input, created_at, updated_at }
  - call_updates: { ownerId, call_id, note, status_change, created_at }

The ownerId ties all data to a specific GridCRM admin account.
"""

import os
import json
import asyncio
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

from groq import AsyncGroq

# ─── Firebase Initialization ─────────────────────────────────────────────────

_firebase_initialized = False


def init_firebase():
    """Initialize Firebase Admin SDK using the GridCRM service account.
    
    Reads credentials from either:
      1. GRIDCRM_FIREBASE_JSON env var (JSON string — used in production/Render)
      2. GRIDCRM_FIREBASE_JSON_PATH env var (file path to the service account JSON)
    """
    global _firebase_initialized
    if _firebase_initialized or firebase_admin._apps:
        _firebase_initialized = True
        return True

    # Option 1: JSON string in env (for Render / Docker deployments)
    json_str = os.environ.get("GRIDCRM_FIREBASE_JSON")
    if json_str:
        try:
            cred_dict = json.loads(json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
            print("[CRM] Firebase initialized from GRIDCRM_FIREBASE_JSON")
            return True
        except Exception as e:
            print(f"[CRM] Error initializing Firebase from JSON: {e}")

    # Option 2: File path to service account JSON
    json_path = os.environ.get("GRIDCRM_FIREBASE_JSON_PATH")
    if json_path and os.path.exists(json_path):
        try:
            cred = credentials.Certificate(json_path)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
            print(f"[CRM] Firebase initialized from {json_path}")
            return True
        except Exception as e:
            print(f"[CRM] Error initializing Firebase from file: {e}")

    print("[CRM] Warning: Firebase credentials not found (set GRIDCRM_FIREBASE_JSON or GRIDCRM_FIREBASE_JSON_PATH)")
    return False


# ─── LLM Extraction ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a data extraction assistant. Given a conversation transcript between a customer support AI and a caller, extract the following fields.

Return ONLY valid JSON with these fields, no markdown, no explanation:
{
  "customer_name": string or null (the caller's name if they mentioned it),
  "call_type": one of ["Service", "Installation", "AMC", "Sales", "Other"],
  "problem_description": string (a clear, concise summary of the customer's problem or requirement),
  "priority": one of ["Low", "Medium", "High"]
}

Rules:
- Extract the customer's name ONLY if they explicitly stated it. Do not invent a name. If none is given, output null.
- Infer call_type accurately:
    - "Service": Customer needs existing equipment repaired, checked, or fixed (e.g. "camera not working", "servicing", "repair", "offline").
    - "Installation": Customer wants new equipment installed or bought (e.g. "want to buy", "new cameras", "need 2 cameras").
    - "AMC": Yearly maintenance contracts.
    - "Sales": General product inquiries without installation context.
- Infer priority from urgency cues ("not working", "urgent", "down" = High; vague requests = Low; default to Medium).
- The problem_description MUST be a clear summary of what the customer needs. If the customer hung up before stating their problem or requirement, you MUST output an empty string "". Do not invent a problem.
- Do not invent information not present in the transcript."""


async def extract_call_data(conversation_history: list[dict]) -> Optional[dict]:
    """Use Groq LLM to extract structured CRM data from the conversation.
    
    Args:
        conversation_history: List of {"role": "user"|"assistant", "content": "..."} dicts
        
    Returns:
        Dict with keys: customer_name, call_type, problem_description, priority
        or None if extraction fails.
    """
    if not conversation_history:
        return None

    # Build a readable transcript from the conversation
    transcript_lines = []
    for msg in conversation_history:
        role = "Customer" if msg["role"] == "user" else "Agent"
        transcript_lines.append(f"{role}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    try:
        async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        response = await async_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Clean markdown fences if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)
        
        # Don't push empty/useless calls (e.g. if they just say "hello" and hang up)
        problem = data.get('problem_description', '')
        if not problem or problem == "No description available" or len(problem) < 5 or data.get('call_type') == 'Other' and not data.get('customer_name'):
             print(f"[CRM] Insufficient data extracted (problem: '{problem}'). Skipping CRM sync.")
             return None

        print(f"[CRM] Extracted: name={data.get('customer_name')}, "
              f"type={data.get('call_type')}, priority={data.get('priority')}, "
              f"problem={problem[:80]}...")
        return data
    except Exception as e:
        if "connect" in str(e).lower() or "timeout" in str(e).lower() or "network" in str(e).lower():
            print(f"[CRM] Network error: Could not reach LLM extraction service")
        else:
            print(f"[CRM] Extraction error: {e.__class__.__name__} - {str(e)}")
        return None


# ─── Firestore Write ──────────────────────────────────────────────────────────

async def push_to_gridcrm(
    conversation_history: list[dict],
    caller_phone: Optional[str] = None,
):
    """Extract data from the call conversation and write it to GridCRM's Firestore.
    
    This function:
      1. Uses LLM to extract customer_name, call_type, problem_description, priority
      2. Uses caller_phone from call metadata
      3. Finds or creates a customer document
      4. Creates a call document
      5. Creates a call_updates audit log entry
      
    All documents use the same schema as GridCRM so they appear natively in the app.
    
    Args:
        conversation_history: The full conversation from the call
        caller_phone: Phone number from call metadata
    """
    # Get ownerId from env — this ties data to your GridCRM admin account
    owner_id = os.environ.get("GRIDCRM_OWNER_ID")
    if not owner_id:
        print("[CRM] Warning: GRIDCRM_OWNER_ID not configured. Skipping CRM sync.")
        return

    if not init_firebase():
        return

    # Step 1: Extract structured data from conversation
    extracted = await extract_call_data(conversation_history)
    if not extracted:
        print("[CRM] Warning: Could not extract data from conversation. Skipping CRM sync.")
        return

    customer_name = extracted.get("customer_name") or "Unknown Customer"
    call_type = extracted.get("call_type", "Service")
    problem_description = extracted.get("problem_description", "No description available")
    priority = extracted.get("priority", "Medium")

    # Clean phone number (digits only)
    phone = None
    if caller_phone:
        phone = "".join(c for c in caller_phone if c.isdigit())
        # Strip leading country code '91' if it's an Indian number with 10 digits after
        if phone.startswith("91") and len(phone) == 12:
            phone = phone[2:]
        elif phone.startswith("0") and len(phone) == 11:
            phone = phone[1:]

    # Build transcript for raw_input field
    transcript_lines = []
    for msg in conversation_history:
        role = "Customer" if msg["role"] == "user" else "Agent"
        transcript_lines.append(f"{role}: {msg['content']}")
    raw_input = "\n".join(transcript_lines)

    try:
        # Run Firestore operations in a thread pool since firebase-admin is synchronous
        await asyncio.get_event_loop().run_in_executor(
            None,
            _write_to_firestore,
            owner_id, customer_name, phone, call_type, problem_description, priority, raw_input
        )
    except Exception as e:
        print(f"[CRM] Error writing to GridCRM: {e}")
        import traceback
        traceback.print_exc()


def _write_to_firestore(
    owner_id: str,
    customer_name: str,
    phone: Optional[str],
    call_type: str,
    problem_description: str,
    priority: str,
    raw_input: str,
):
    """Synchronous Firestore write (called from thread pool).
    
    Matches GridCRM's exact schema from routes.py create_call action (lines 361-415).
    """
    db = firestore.client()

    # Step 2: Find or create customer
    customer_id = None
    customer_data = {"name": customer_name, "phone": phone or ""}

    if phone:
        # Try to find existing customer by phone number (same logic as GridCRM routes.py line 368)
        existing = db.collection("customers").where("ownerId", "==", owner_id).where("phone", "==", phone).get()
        if existing:
            customer_doc = existing[0]
            customer_id = customer_doc.id
            customer_data = customer_doc.to_dict()
            # Update name if we got a real name and existing is "Unknown"
            if customer_name != "Unknown Customer" and customer_data.get("name") in (None, "", "Unknown Customer", "Unknown"):
                customer_doc.reference.update({"name": customer_name})
                customer_data["name"] = customer_name
            print(f"[CRM] Matched existing customer: {customer_data.get('name')} ({phone})")

    if not customer_id:
        # Create new customer (same schema as GridCRM routes.py lines 376-383)
        _, new_ref = db.collection("customers").add({
            "ownerId": owner_id,
            "name": customer_name,
            "phone": phone or "",
            "address": "",
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        customer_id = new_ref.id
        print(f"[CRM] Created new customer record: {customer_name} ({phone or 'no phone'})")

    # Step 3: Create call document (same schema as GridCRM routes.py lines 387-403)
    call_ref = db.collection("calls").document()
    call_data = {
        "ownerId": owner_id,
        "customer_id": customer_id,
        "customer": {
            "name": customer_data.get("name", customer_name),
            "phone": customer_data.get("phone", phone or ""),
        },
        "call_type": call_type,
        "problem_description": problem_description,
        "priority": priority,
        "status": "Pending",
        "technician_assigned": None,
        "raw_input": raw_input,
        "source": "voicera",  # Tag so you know this came from Voicera
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }
    call_ref.set(call_data)
    print(f"[CRM] Created call ticket: {customer_name} | {call_type} | {priority} priority")

    # Step 4: Create audit log entry (same as GridCRM routes.py lines 405-411)
    db.collection("call_updates").add({
        "ownerId": owner_id,
        "call_id": call_ref.id,
        "note": "Voicera: Call auto-logged from inbound phone call.",
        "status_change": "Pending",
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    print(f"[CRM] Call successfully synced to GridCRM (call_id={call_ref.id})")
