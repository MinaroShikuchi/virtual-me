import re

with open('/home/minaro/Git/virtual-me/dissussion.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output = []
in_chatgpt = False
buffer = []

def flush_buffer():
    global buffer
    if buffer:
        condensed = ' '.join(x.strip() for x in buffer if x.strip())
        output.append(condensed + '\n\n')
        buffer.clear()

for line in lines:
    if line.strip() == 'You said:':
        if in_chatgpt:
            flush_buffer()
            in_chatgpt = False
        output.append(line)
    elif line.strip() == 'ChatGPT said:':
        output.append(line)
        in_chatgpt = True
    else:
        if in_chatgpt:
            buffer.append(line)
        else:
            output.append(line)

if in_chatgpt:
    flush_buffer()

with open('/home/minaro/Git/virtual-me/dissussion.md', 'w', encoding='utf-8') as f:
    for line in output:
        f.write(line)
