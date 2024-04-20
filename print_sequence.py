import pickle
import sys

path = sys.argv[1]
with open(path, 'rb') as f:
    messages = pickle.load(f)

for i, m in enumerate(messages):
    print("# Human" if i % 2 == 0 else "# AI", i // 2)
    print("---")
    if not isinstance(m, dict):
        m = m.__dict__
    print(m['content'])

