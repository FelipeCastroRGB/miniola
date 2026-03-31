from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "miniola_py" / "process.py"), run_name="__main__")
