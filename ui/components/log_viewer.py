"""
ui/components/log_viewer.py — Shared scrollable log viewer with auto-follow.

Usage:
    from ui.components.log_viewer import scrollable_log

    log_box = st.empty()
    scrollable_log(log_box, lines)          # auto-follows by default
    scrollable_log(log_box, lines, follow=False)  # static view
"""
from __future__ import annotations

import html as _html

import streamlit.components.v1 as _components


def scrollable_log(
    container,
    lines: list[str],
    max_height: int = 300,
    follow: bool = True,
    title: str = "Log Output",
) -> None:
    """Render *lines* inside a scrollable, auto-following log viewer.

    Parameters
    ----------
    container
        A Streamlit placeholder (``st.empty()``) that will hold the component.
    lines
        Log lines to display (oldest first).
    max_height
        Maximum height of the scrollable area in pixels.
    follow
        If ``True`` (default), the viewer auto-scrolls to the bottom on each
        render.  The user can scroll up to pause auto-follow; a floating
        "↓ Follow" button appears to re-enable it.
    title
        Label shown in the header bar.
    """
    if not lines:
        container.empty()
        return

    # HTML-escape every line, then join with <br> for the pre-wrap container
    escaped = "<br>".join(
        _html.escape(ln) for ln in lines
    )

    count = len(lines)
    auto_follow_js = "true" if follow else "false"

    # The entire component is a self-contained HTML document rendered inside
    # an iframe via st.components.v1.html.  All interactivity (scroll
    # detection, follow toggle) is handled in JS — no Streamlit round-trips.
    component_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}

  .log-wrapper {{
    border: 1px solid #333;
    border-radius: 6px;
    overflow: hidden;
    background: #0e1117;
    position: relative;
  }}

  .log-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    background: #161b22;
    border-bottom: 1px solid #333;
    font-size: 12px;
    color: #8b949e;
  }}
  .log-header .title {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
  }}
  .log-header .count {{
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 11px;
    color: #8b949e;
  }}

  .log-body {{
    max-height: {max_height}px;
    overflow-y: auto;
    padding: 10px 12px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', Menlo, Consolas, monospace;
    font-size: 12.5px;
    line-height: 1.55;
    color: #c9d1d9;
    white-space: pre-wrap;
    word-break: break-word;
    /* Hide until JS positions the scroll, then reveal — prevents the
       flash-to-top that occurs when the iframe is re-created. */
    visibility: hidden;
  }}
  .log-body.ready {{
    visibility: visible;
  }}

  /* Scrollbar styling */
  .log-body::-webkit-scrollbar {{ width: 6px; }}
  .log-body::-webkit-scrollbar-track {{ background: transparent; }}
  .log-body::-webkit-scrollbar-thumb {{
    background: #30363d;
    border-radius: 3px;
  }}
  .log-body::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}

  /* Follow button */
  .follow-btn {{
    position: absolute;
    bottom: 12px;
    right: 16px;
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 16px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: none;
    align-items: center;
    gap: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    transition: background 0.15s, transform 0.1s;
    z-index: 10;
  }}
  .follow-btn:hover {{
    background: #2ea043;
    transform: translateY(-1px);
  }}
  .follow-btn.visible {{
    display: inline-flex;
  }}
</style>
</head>
<body>
  <div class="log-wrapper">
    <div class="log-header">
      <span class="title">📋 {_html.escape(title)}</span>
      <span class="count">{count} line{"s" if count != 1 else ""}</span>
    </div>
    <div class="log-body" id="logBody">{escaped}</div>
    <button class="follow-btn" id="followBtn" title="Resume auto-follow">↓ Follow</button>
  </div>

  <script>
    (function() {{
      const body = document.getElementById('logBody');
      const btn  = document.getElementById('followBtn');
      let autoFollow = {auto_follow_js};
      const THRESHOLD = 40;  // px from bottom to consider "at bottom"

      function isAtBottom() {{
        return (body.scrollHeight - body.scrollTop - body.clientHeight) < THRESHOLD;
      }}

      function scrollToBottom() {{
        body.scrollTop = body.scrollHeight;
      }}

      // On scroll: detect if user scrolled away from bottom
      body.addEventListener('scroll', function() {{
        if (isAtBottom()) {{
          autoFollow = true;
          btn.classList.remove('visible');
        }} else {{
          autoFollow = false;
          btn.classList.add('visible');
        }}
      }});

      // Follow button click
      btn.addEventListener('click', function() {{
        autoFollow = true;
        btn.classList.remove('visible');
        scrollToBottom();
      }});

      // Initial scroll — position first, then reveal to avoid flash
      if (autoFollow) {{
        scrollToBottom();
      }} else if (!isAtBottom()) {{
        btn.classList.add('visible');
      }}
      // Reveal the log body now that scroll is positioned
      body.classList.add('ready');
    }})();
  </script>
</body>
</html>
"""

    # +50 accounts for the header bar, borders, and padding
    frame_height = max_height + 50

    with container:
        _components.html(component_html, height=frame_height, scrolling=False)
