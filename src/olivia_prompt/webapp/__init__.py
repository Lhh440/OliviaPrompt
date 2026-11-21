"""Web app — stdlib server + no-build single page app.

Deliberately NOT FastAPI: the product promise is "double-click and use", so the
web tier must not add an install burden.  ``ThreadingHTTPServer`` + vanilla JS
keeps `olivia-prompt web` a zero-dependency command.
"""

from .server import make_server, serve

__all__ = ["make_server", "serve"]
