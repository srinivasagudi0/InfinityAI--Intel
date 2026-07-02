from brain.modes import get_mode
from brain.rules import get_global_rules


def build_prompt(
    user_message,
    mode_name,
    history,
    memories=None,
    file_context=None,
    agent_instructions=None,
):

    mode = get_mode(mode_name)
    rules = get_global_rules()
    memories = memories or []

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

    if agent_instructions:
        system_prompt += f"\n\nSELECTED AGENT INSTRUCTIONS:\n{agent_instructions}"

    if memories:
        memory_lines = "\n".join(f"- {memory}" for memory in memories)
        system_prompt += f"\n\nPERSISTENT USER MEMORY:\n{memory_lines}"

    if file_context:
        system_prompt += f"\n\nATTACHED FILE CONTEXT:\n{file_context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    return messages
