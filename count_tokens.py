import sys

with open('/home/minaro/Git/virtual-me/dissussion.md', 'r', encoding='utf-8') as f:
    text = f.read()
chars = len(text)
words = len(text.split())
print(f"Characters: {chars}")
print(f"Words: {words}")
print(f"Estimated Gemini Tokens (1 token ~= 4 chars): {chars / 4:.0f}")
print(f"Estimated Gemini Tokens (1 token ~= 0.75 words): {words / 0.75:.0f}")
