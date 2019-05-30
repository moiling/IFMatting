import os

ref_dir = os.path.join(os.path.dirname(__file__))

__all__ = [f[:-3] for f in os.listdir(ref_dir) if f.endswith('.py') and
           not f.startswith('__')]
__all__.sort()

for f in __all__:
    __import__(__name__ + '.' + f)

del f, ref_dir
