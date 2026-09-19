"""Placed at the project root so pytest puts the root on sys.path.

That is what lets tests import `src.config` / `src.schema` by their real package
paths, rather than tests needing to fiddle with sys.path themselves.
"""
