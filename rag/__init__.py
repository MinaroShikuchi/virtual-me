"""rag package — retrieval-augmented generation components."""
import logging

# Configure rag.* loggers: INFO shows [AGENT]/[TOOL] progress,
# DEBUG (opt-in) shows verbose intent router / purification output.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_log = logging.getLogger(__name__)          # "rag"
_log.addHandler(_handler)
_log.setLevel(logging.INFO)
