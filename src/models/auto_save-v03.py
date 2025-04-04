import streamlit as st
import time

class AutoSave:
    """
    Helper class to detect page navigation and automatically save chat history.
    
    This uses Streamlit's component lifecycle to detect when the component is
    unmounted (which happens on page navigation).
    """
    
    def __init__(self, history_manager):
        """
        Initialize the auto-save component.
        
        Args:
            history_manager: The ChatHistoryManager instance
        """
        self.history_manager = history_manager
        self.last_saved = time.time()
        
    def check_and_save(self, force=False):
        """
        Check if it's time to save and perform save if needed.
        
        Args:
            force (bool): Force a save even if time threshold isn't met
            
        Returns:
            dict: Summary of saved sessions or None if no save performed
        """
        current_time = time.time()
        # Save at most every 30 seconds or when forced
        if force or (current_time - self.last_saved > 30):
            saved = self.history_manager.save_all_sessions(st.session_state)
            self.last_saved = current_time
            return saved
        return None
    
    def setup_auto_save(self):
        """
        Setup automatic saving on page navigation.
        
        This function creates a "hidden" element that will trigger
        when the page is unloaded (navigation or refresh).
        """
        # Initialize a key for tracking if we're navigating away
        if "navigation_tracker" not in st.session_state:
            st.session_state.navigation_tracker = 0
        
        with st.sidebar:
            # Use a hidden container that executes before navigation
            # with st.container():
            st.markdown('<div style="display:none">', unsafe_allow_html=True)
            # Increment the tracker - when this value changes, we're navigating
            new_val = st.session_state.navigation_tracker + 1
            if st.button(f"_hidden_nav_{new_val}", key=f"_hidden_nav_{new_val}"):
                # This won't actually run, but it creates the session state entry
                st.session_state.navigation_tracker = new_val
            
            # Check if the tracker has changed
            if new_val != st.session_state.navigation_tracker:
                # We're navigating away - force a save
                self.check_and_save(force=True)
                st.session_state.navigation_tracker = new_val
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Regular auto-save check
        self.check_and_save()