class EMA:
    """
    Exponential moving average
    Initialize EMA with a smoothing factor. Lower alpha = smoother changes, higher = EMA more responsive to recent changes.
    EMA prevents harsh changed in rotation harming a real robot
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        #checks if value first value
        if self.value is None:
            self.value = new_value
        #blends new value with average, more weight to recent values
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value