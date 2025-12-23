
"""
Text progress monitor for agent progress display.
"""
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from tqdm import tqdm

# pylint: disable=too-many-instance-attributes

class TextProgressMonitor:
    """Text progress monitor, suitable for scenarios with only textual description and no percentage."""

    def __init__(self,
                 connection,
                 connection_id: int,
                 refresh_interval: float = 1.0,
                 show_progress: bool = True):
        """
        Initialize the text progress monitor.

        Parameters
        ----------
        connection : database connection object
            The database connection used to query progress information.
        connection_id : int
            The current connection ID, used to query corresponding progress information.
        refresh_interval : float, optional
            Progress query interval (seconds), default is 1.0 second.
        show_progress : bool, optional
            Whether to display progress, default is True.
        """
        self.connection = connection
        self.connection_id = connection_id
        self.refresh_interval = refresh_interval
        self.show_progress = show_progress
        # Progress related
        self.progress_messages: List[Dict[str, str]] = []
        self.last_message = ""
        self.last_displayed_message = ""
        # Monitor state
        self.is_monitoring = False
        self.completed = False
        # Progress callbacks
        self.on_progress_update: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[bool], None]] = None
        # tqdm progress bar
        self.progress_bar = None
        self.message_count = 0

    def start(self) -> 'TextProgressMonitor':
        """Start progress monitoring."""
        if self.show_progress:
            # Use custom format, only show elapsed time and message
            self.progress_bar = tqdm(
                total=None,  # No definite total progress
                bar_format="{n}[{elapsed}{postfix}]",  # Show elapsed time and description
                dynamic_ncols=True,
                leave=True
            )
            # Set initial description
            self.progress_bar.set_postfix_str("Starting...")
        self.is_monitoring = True
        self.completed = False
        self.start_time = datetime.now()
        self.message_count = 0
        return self

    def update(self) -> Optional[str]:
        """
        Update progress information.

        Returns
        -------
        str or None
            Returns the latest progress message, or None if there is no new message.
        """
        if not self.is_monitoring or self.completed:
            return None
        try:
            # Only query PROGRESS_DETAIL field
            progress_sql = f"""
                SELECT PROGRESS_DETAIL, CURRENT_PROGRESS
                FROM M_JOB_PROGRESS
                WHERE CONNECTION_ID = {self.connection_id}
                ORDER BY START_TIME DESC
                LIMIT 1
            """
            with self.connection.cursor() as cursor:
                cursor.execute(progress_sql)
                result = cursor.fetchone()
                if result and result[0]:
                    current_message = result[0]
                    current_progress = result[1]
                    # If there is a new message
                    if current_message and current_message != self.last_message:
                        self.last_message = current_message
                        self.message_count += 1

                        # Use local time as timestamp
                        timestamp = datetime.now().strftime("%H:%M:%S")

                        self.progress_messages.append({
                            'timestamp': timestamp,
                            'message': current_message
                        })

                        # Use tqdm to display progress
                        if self.show_progress and current_message != self.last_displayed_message:
                            if self.progress_bar is not None:
                                # Directly set description, bar_format will show elapsed time
                                self.progress_bar.set_postfix_str(current_message)
                                self.progress_bar.n = current_progress
                                self.progress_bar.refresh()
                            self.last_displayed_message = current_message

                        # Call progress callback function
                        if self.on_progress_update:
                            self.on_progress_update(current_message)

                        return current_message
                else:
                    # No progress information
                    if self.show_progress and self.progress_bar is not None:
                        if self.message_count == 0:
                            self.progress_bar.set_postfix_str("Waiting for progress update...")
        except Exception as exc:
            # If progress query fails, do not interrupt main process
            if self.show_progress and self.progress_bar is not None:
                error_msg = str(exc)[:50]
                self.progress_bar.set_postfix_str("Error: %s..." % error_msg)

        return None

    def complete(self, success: bool = True, final_message: str = None) -> None:
        """
        Complete progress monitoring.

        Parameters
        ----------
        success : bool, optional
            Whether completed successfully, default is True.
        final_message : str, optional
            Final message to display.
        """
        self.completed = True
        if self.show_progress and self.progress_bar is not None:
            end_time = datetime.now()
            elapsed_time = end_time - self.start_time
            elapsed_seconds = elapsed_time.total_seconds()
            elapsed_str = self._format_time(elapsed_seconds)

            # Update tqdm with final status
            if final_message:
                self.progress_bar.set_postfix_str(final_message)
            else:
                if success:
                    self.progress_bar.set_postfix_str("Query completed successfully")
                else:
                    self.progress_bar.set_postfix_str("Query failed")

            # Refresh display
            self.progress_bar.refresh()

            # Close the progress bar
            time.sleep(0.1)  # Short delay to ensure display
            self.progress_bar.close()

        # Call complete callback
        if self.on_complete:
            self.on_complete(success)
        self.is_monitoring = False

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format elapsed time for display."""
        if seconds < 60:
            return "%ds" % int(seconds)
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return "%dm%ds" % (minutes, secs)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return "%dh%dm" % (hours, minutes)

    def stop(self) -> None:
        """Stop progress monitoring."""
        if not self.completed:
            if self.show_progress and self.progress_bar is not None:
                self.progress_bar.set_postfix_str("Stopped")
                time.sleep(0.1)
                self.progress_bar.close()
            self.completed = True
        self.is_monitoring = False

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, str]]:
        """Get recent progress messages."""
        return self.progress_messages[-count:] if self.progress_messages else []

    def get_all_messages(self) -> List[Dict[str, str]]:
        """Get all progress messages."""
        return self.progress_messages.copy()

    def clear_messages(self) -> None:
        """Clear progress messages."""
        self.progress_messages = []

    def register_callback(self,
                         on_progress: Optional[Callable[[str], None]] = None,
                         on_complete: Optional[Callable[[bool], None]] = None) -> None:
        """
        Register callback functions.
        Parameters
        ----------
        on_progress : Callable[[str], None], optional
            Callback function when progress updates.
        on_complete : Callable[[bool], None], optional
            Callback function when completed.
        """
        self.on_progress_update = on_progress
        self.on_complete = on_complete
