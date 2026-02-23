"""
virtual_me/pages/chat.py — Chat page.
"""
import reflex as rx
from virtual_me.state.chat_state import ChatState, ChatMessage
from virtual_me.components.layout import layout


def _token_bar(msg: ChatMessage) -> rx.Component:
    """Token usage display."""
    return rx.cond(
        msg.prompt_tokens > 0,
        rx.vstack(
            rx.hstack(
                rx.text("prompt: ", size="1", color="#64748b"),
                rx.text(msg.prompt_tokens.to(str), size="1", color="#64748b"),
                rx.text(" | response: ", size="1", color="#64748b"),
                rx.text(msg.completion_tokens.to(str), size="1", color="#64748b"),
                spacing="0",
            ),
            spacing="1",
            width="100%",
        ),
        rx.fragment(),
    )


def _thinking_block(msg: ChatMessage) -> rx.Component:
    """Expandable thinking block."""
    return rx.cond(
        msg.thinking != "",
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text("Thinking…", size="2", color="#94a3b8"),
                content=rx.box(
                    rx.el.pre(
                        msg.thinking,
                        style={
                            "white_space": "pre-wrap",
                            "color": "#94a3b8",
                            "font_size": "0.82rem",
                        },
                    ),
                    bg="#181b2e",
                    border_left="3px solid #6366f1",
                    padding="10px 14px",
                    border_radius="0 8px 8px 0",
                ),
                value="thinking",
            ),
            type="multiple",
            width="100%",
        ),
        rx.fragment(),
    )


def _intent_block(msg: ChatMessage) -> rx.Component:
    """Intent and facts expandable block."""
    return rx.cond(
        msg.facts.length() > 0,
        rx.accordion.root(
            rx.accordion.item(
                header=rx.hstack(
                    rx.text("Intent & Facts (", size="2"),
                    rx.text(msg.facts.length().to(str), size="2"),
                    rx.text(" found)", size="2"),
                    spacing="0",
                ),
                content=rx.vstack(
                    rx.foreach(
                        msg.facts,
                        lambda f: rx.hstack(
                            rx.text("• ", size="2", color="#e2e8f0"),
                            rx.text(f, size="2", color="#e2e8f0"),
                            spacing="0",
                        ),
                    ),
                    spacing="1",
                ),
                value="facts",
            ),
            type="multiple",
            width="100%",
        ),
        rx.fragment(),
    )


def _deliberation_item(d: dict) -> rx.Component:
    """Render a single deliberation entry."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(d["persona"], weight="bold", size="2"),
                rx.text(" (Round ", weight="bold", size="2"),
                rx.text(d["round"].to(str), weight="bold", size="2"),
                rx.text(")", weight="bold", size="2"),
                spacing="0",
            ),
            rx.text(d["response"], size="2", color="#94a3b8"),
            spacing="1",
        ),
        padding="8px",
        border="1px solid #2d3250",
        border_radius="8px",
        margin_bottom="4px",
    )


def _deliberation_block(msg: ChatMessage) -> rx.Component:
    """Deliberation committee expandable block."""
    return rx.cond(
        msg.deliberations.length() > 0,
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text("Inner Deliberation Committee", size="2"),
                content=rx.vstack(
                    rx.foreach(msg.deliberations, _deliberation_item),
                    spacing="2",
                ),
                value="deliberations",
            ),
            type="multiple",
            width="100%",
        ),
        rx.fragment(),
    )


def _message_bubble(msg: ChatMessage) -> rx.Component:
    """Render a single chat message."""
    return rx.cond(
        msg.role == "user",
        # User message
        rx.box(
            rx.text(msg.content, color="#f1f5f9"),
            bg="#2d3250",
            padding="12px 16px",
            border_radius="12px",
            max_width="80%",
            align_self="flex-end",
        ),
        # Assistant message
        rx.box(
            rx.vstack(
                _thinking_block(msg),
                _intent_block(msg),
                _deliberation_block(msg),
                rx.markdown(msg.content),
                _token_bar(msg),
                spacing="3",
                width="100%",
            ),
            bg="#1e2030",
            padding="12px 16px",
            border_radius="12px",
            max_width="90%",
            align_self="flex-start",
            border="1px solid #2d3250",
        ),
    )


def chat_content() -> rx.Component:
    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Chat with your memories", size="6"),
            rx.spacer(),
            rx.button(
                "Clear",
                variant="ghost",
                size="1",
                on_click=ChatState.clear_history,
            ),
            width="100%",
            align="center",
        ),

        # Committee status
        rx.cond(
            ChatState.active_personas.length() > 0,
            rx.hstack(
                rx.text("Committee Active: ", size="1", color="#94a3b8"),
                rx.text(ChatState.deliberation_rounds.to(str), size="1", color="#94a3b8"),
                rx.text(" rounds", size="1", color="#94a3b8"),
                spacing="0",
            ),
            rx.fragment(),
        ),

        # Messages
        rx.box(
            rx.vstack(
                rx.foreach(ChatState.messages, _message_bubble),
                spacing="3",
                width="100%",
            ),
            flex="1",
            overflow_y="auto",
            width="100%",
            padding="8px",
        ),

        # Loading indicator
        rx.cond(
            ChatState.is_loading,
            rx.hstack(
                rx.spinner(size="2"),
                rx.text(ChatState.loading_status, size="2", color="#94a3b8"),
                spacing="2",
                align="center",
            ),
            rx.fragment(),
        ),

        # Error
        rx.cond(
            ChatState.error_message != "",
            rx.callout(
                ChatState.error_message,
                icon="circle-alert",
                color_scheme="red",
            ),
            rx.fragment(),
        ),

        # Input
        rx.hstack(
            rx.input(
                placeholder="Ask anything about your memories…",
                value=ChatState.current_input,
                on_change=ChatState.set_current_input,
                width="100%",
                size="3",
            ),
            rx.button(
                rx.icon("send", size=18),
                on_click=ChatState.send_message,
                loading=ChatState.is_loading,
                color_scheme="iris",
                size="3",
            ),
            spacing="2",
            width="100%",
        ),

        spacing="4",
        height="calc(100vh - 48px)",
        width="100%",
    )


def chat_page() -> rx.Component:
    return layout(chat_content())
