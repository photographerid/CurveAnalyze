# Windows EXE Build

## Scope
This project should be built as a Windows `--onedir` PyInstaller app first. Do not move to an installer until the packaged app has been tested on Windows.

## Why `--onedir` first
- `LICENSE` and `README.md` are bundled as companion resources.
- The app stores user-specific recent-project history in a per-user writable directory.
- `--onedir` is the simplest path to validate runtime behavior before introducing installer work.

## Build Steps
1. Copy the project to a Windows machine.
2. Open `cmd` or PowerShell in the project root.
3. Create and activate a virtual environment:

```bat
py -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Verify the source app runs:

```bat
python CurveAnalyze.py
```

6. Build the app:

```bat
pyinstaller --noconfirm --windowed --onedir --name CurveAnalyze --add-data "LICENSE;." --add-data "README.md;." CurveAnalyze.py
```

7. Run the packaged app:

```text
dist\CurveAnalyze\CurveAnalyze.exe
```

## Test Checklist
- App launches without a console window
- CSV import works
- Manual entry works
- 3PL and 4PL fitting work
- Single and bulk OD estimation work
- Plot export works
- Save and load project work
- `New Project` works
- Unsaved-change prompts work
- `Recent Projects` works
- Help > License works
- Help > User Guide works

## Notes
- There is currently no `icon.ico` in the repository. The app will still run without a custom icon.
- Recent-project history is stored outside the install directory in a per-user location.
- Distribute the entire `dist\CurveAnalyze\` folder for this first Windows test, not only the `.exe`.
