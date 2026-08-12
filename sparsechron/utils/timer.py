"""Timer utility module for tracking execution time."""

import time
class Timer:
    """A simple timer for tracking elapsed time."""
    
    def __init__(self) -> None:
        """Initializes the timer and starts it."""
        self.start_time = time.time()
        
    def elapsed_minutes(self) -> float:
        """Returns the elapsed time in minutes since initialization.
        
        Returns:
            The elapsed time in minutes.
        """
        return (time.time() - self.start_time) / 60.0
    
    def reset(self) -> None:
        """Resets the timer to the current time."""
        self.start_time = time.time()
