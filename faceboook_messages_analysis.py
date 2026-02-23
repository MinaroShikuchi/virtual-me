import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

tqdm.pandas()

def analyze_message_distribution(data):
    # Convert to dataframe and sort
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['conversation', 'date'])

    # Calculate gap in minutes
    df['gap_mins'] = df.groupby('conversation')['date'].diff().dt.total_seconds() / 60
    
    # Drop NaN (first message of every conversation)
    gaps = df['gap_mins'].dropna()

    # Define common thresholds to check (in minutes)
    thresholds = [1, 5, 10, 15, 30, 45, 60, 120, 240, 480, 720, 1440, 2880, 4320, 4800, 5040, 5160, 5280, 5760, 7200, 8640]
    print("--- Message Gap Analysis (Console Stats) ---")
    print(f"Total Messages: {len(df):,}")
    print(f"Mean Gap: {gaps.mean():.2f} mins")
    print(f"Median Gap: {gaps.median():.2f} mins")
    print("\nPercentile Breakdown:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(gaps, p):.2f} mins | {np.percentile(gaps, p)/60:.2f} hours")

    print("\n--- Potential Session Breakpoints ---")
    print("If you set the threshold to X, here is how many 'sessions' you get:")
    
    for t in thresholds:
        # A new session is created every time the gap > threshold
        num_sessions = (gaps > t).sum() + df['conversation'].nunique()
        avg_msgs_per_session = len(df) / num_sessions
        
        label = f"{t}m" if t < 60 else f"{t/60}h"
        if t == 1440: label = "24h"
        
        print(f"Threshold {label: >4}: {num_sessions:,} sessions | Avg {avg_msgs_per_session:.1f} msgs/session")
def group_messages_for_chroma(data, base_threshold=60, overlap=2):
    """
    First Pass: Groups messages using a dynamic threshold based on local tempo.
    """
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    # Reset index to enable fast O(1) context lookups
    df = df.sort_values(['conversation', 'date']).reset_index(drop=True)

    # 1. Calculate gaps between messages in minutes
    df['gap'] = df.groupby('conversation')['date'].diff().dt.total_seconds() / 60
    
    # 2. LOCAL Velocity Logic
    df['local_tempo'] = df.groupby('conversation')['gap'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    def is_new_session(row):
        if pd.isna(row['gap']): return True 
        if row['gap'] > (row['local_tempo'] * 5) and row['gap'] > 15:
            return True
        if row['gap'] > base_threshold:
            return True
        return False

    print("Calculating session boundaries...")
    tqdm.pandas(desc="Session boundaries")
    df['new_session_gap'] = df.progress_apply(is_new_session, axis=1)
    df['session_id'] = df.groupby('conversation')['new_session_gap'].cumsum()

    def format_block(group):
        idx = group.index
        conv_id = group['conversation'].iloc[0]
        
        # Pull overlapping context (Optimized O(1) lookup instead of O(N) scan)
        start_idx = max(0, idx[0] - overlap)
        prev_msgs = df.iloc[start_idx:idx[0]]
        prev_msgs = prev_msgs[prev_msgs['conversation'] == conv_id]
        
        prefix = "\n".join([f"{r['sender_name']}: {r['text']} [CONTEXT]" for _, r in prev_msgs.iterrows()])
        
        current_text = "\n".join([f"{row['sender_name']}: {row['text']}" for _, row in group.iterrows()])
        full_display_text = f"{prefix}\n{current_text}".strip()
        
        return pd.Series({
            'text': full_display_text,
            'raw_content': current_text, # Used for enrichment
            'start_time': group['date'].min().isoformat(),
            'end_time': group['date'].max().isoformat(),
            'msg_count': len(group),
            'conversation_id': conv_id,
            'senders': list(group['sender_name'].unique()),
            'source': 'facebook'
        })

    print("Formatting message blocks...")
    groups = [group for _, group in df.groupby(['conversation', 'session_id'])]
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(format_block, groups), total=len(groups), desc="Formatting blocks"))
        
    return pd.DataFrame(results)

df = pd.read_json('facebook_messages.json')
analyze_message_distribution(df)

df = group_messages_for_chroma(df)
df.to_json('facebook_messages_grouped.json', orient='records', lines=True)
print(df.head())