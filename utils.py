class CallCountDecorator:
    """
    A decorator that will count and print how many times the decorated function was called
    """

    def __init__(self, inline_func):
        self.call_count = 0
        self.inline_func = inline_func

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self._print_call_count()
        return self.inline_func(*args, **kwargs)

    def _print_call_count(self):
        print(f"The {self.inline_func.__name__} called {self.call_count} times")
