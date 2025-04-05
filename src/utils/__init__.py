# utils/__init__.py
# This file makes the utils directory a Python package
# Import common utilities here to make them available from the utils package

from .ui_helpers import (
    initialize_session_state,
    display_chat_messages,
    handle_chat_input,
    process_chat_message,
    add_user_message,
    save_chat_history,
    display_example_questions,
    add_assistant_message,
    display_history_view
)

# Import debugging utilities
from .debugging import (
    display_debug_page_header,
    compare_retrieval_parameters,
    display_retrieval_comparison,
    analyze_document_chunks,
    run_rag_evaluation,
    display_evaluation_results,
    generate_test_questions,
    export_debug_report
)

# Import visualization utilities
from .visualization import (
    generate_visualization,
    generate_visualization_code,
    suggest_visualizations,
    format_sql_for_display
)

# You can import other utility modules as needed
# from .sidebar import display_sidebar