from pathlib import Path
import os


for i in range(15):
    f_name = Path("problems", f"problem{i}.json")
    new_name = Path("problems", f"problem-{i}.json")
    if os.path.isfile(f_name):
        os.rename(f_name, new_name)
