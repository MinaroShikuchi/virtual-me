from pathlib import Path

dash_path = Path('ui/dashboard.py')
lines = dash_path.read_text().split('\n')

# Find "if alive:" and "else:"
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "if alive:":
        start_idx = i
    elif line.strip() == "else:" and "st.info(\"Connect to Neo4j to view knowledge graph stats.\")" in lines[i+1]:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    for i in range(start_idx + 1, end_idx):
        if lines[i].strip() != "":
            lines[i] = "    " + lines[i]
    lines[end_idx] = "    else:"
    # ensure st.info is indented
    lines[end_idx+1] = "        " + lines[end_idx+1].lstrip()

dash_path.write_text('\n'.join(lines))
