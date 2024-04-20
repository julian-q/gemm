import os

def run_in_tmux(session_name, cmd):
    os.system(f"tmux new-session -d -s {session_name}")
    os.system(f"tmux send-keys -t {session_name} '{cmd}' C-m")

base_cmd = lambda name: f"python e2e.py --n_instances 16 --n_steps 16 --run_name {name}"

# Commands
commands = [
    f"{base_cmd('openai')}        --provider OPENAI --model gpt-4-turbo-preview --run_idx 0",
    f"{base_cmd('openai_w_docs')} --provider OPENAI --model gpt-4-turbo-preview --run_idx 1 --use_docs",
    f"{base_cmd('claude')}        --provider ANTHROPIC --model claude-3-opus-20240229 --run_idx 2",
    f"{base_cmd('claude_w_docs')} --provider ANTHROPIC --model claude-3-opus-20240229 --run_idx 3 --use_docs",
    f"{base_cmd('gemini')}        --provider GOOGLE --model gemini-1.0-pro --run_idx 4",
    f"{base_cmd('gemini_w_docs')} --provider GOOGLE --model gemini-1.0-pro --run_idx 5 --use_docs"
]

# Run each command in a separate tmux session
for i, cmd in enumerate(commands):
    session_name = f"session_{i}"
    run_in_tmux(session_name, cmd)

