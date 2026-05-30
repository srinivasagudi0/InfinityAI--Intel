from brain.modes import get_mode
from brain.rules import get_global_rules

def build_prompt(user_message, mode_name, history):

    mode = get_mode(mode_name)
    rules = get_global_rules()

    # simple system prompt to get eveythig together, is kind of cool instead of clutter

    system_prompt = f"""
    {rules}

    ---

    Mode: {mode['name'].upper()}
    Tone: {mode['tone']}

    {mode['system_prompt']}

    YOUR INSTRUCTIONS FOR THIS RESPONSE:
    {mode['instructions']}
    """.strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    return messages
