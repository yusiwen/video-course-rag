"""video-course-rag — turn lecture recordings into a self-contained course package.

The pipeline is file-based: one directory per recorded run, and every stage reads
and writes only inside that directory. Nothing is stored in a database, so a run
can be inspected, copied, re-run or deleted as a unit.
"""
__version__ = "0.1.0a1"
