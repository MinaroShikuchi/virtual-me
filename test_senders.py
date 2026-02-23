import pandas as pd
df = pd.read_json('facebook_messages_grouped.json', lines=True)
print("Total groups:", len(df))
print("Groups with only 1 sender:", df[df['senders'].apply(len) == 1].shape[0])
print("\nSample of groups with 1 sender (msg_count > 1):")
print(df[(df['senders'].apply(len) == 1) & (df['msg_count'] > 1)][['conversation_id', 'msg_count', 'senders', 'raw_content']].head())

print("\nSample of groups with >1 sender:")
print(df[df['senders'].apply(len) > 1][['conversation_id', 'msg_count', 'senders', 'raw_content']].head())
