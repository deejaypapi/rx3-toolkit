import pathlib
import subprocess
import sys


# Windows: prevent child console processes from creating black windows.
if sys.platform == "win32":
    _OriginalPopen = subprocess.Popen

    class _NoConsolePopen(_OriginalPopen):
        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = (
                kwargs.get("creationflags", 0)
                | subprocess.CREATE_NO_WINDOW
            )
            super().__init__(*args, **kwargs)

    subprocess.Popen = _NoConsolePopen


import tkinter as tk

repository = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import stem_studio
import theme

root = tk.Tk()
theme.apply(root)
root.title("RX3 Stem Studio")
app = stem_studio.StemStudioPane(root)
app.pack(fill="both", expand=True)
root.mainloop()