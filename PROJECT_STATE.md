# Project State

## Project Goal
`CurveAnalyze.py` is a Tkinter desktop application for ELISA standard-curve analysis. The goal is to provide a lightweight GUI for:
- entering or importing standard concentrations and OD values
- fitting 4PL or 3PL logistic curves
- estimating unknown concentrations from OD values
- handling bulk OD estimation
- saving/loading project sessions
- exporting plots and bulk results

The app is being prepared for eventual Windows packaging/installer distribution, but current work has focused on making the Linux-side source app functionally solid first.

## Current Status
The app is functionally in good shape and has been iteratively refined. Current implemented capabilities include:

- CSV import with concentration/OD column detection
- Manual standard entry
- 4PL and 3PL fitting
- Support for both increasing and decreasing data trends
- Single OD concentration estimation
- Bulk OD estimation
- Export of the standard-curve plot
- Export of bulk estimation results to CSV
- Save / Save As / Load project using JSON-based project files
- `New Project`
- unsaved-work prompting on:
  - app close
  - `Load Project`
  - `New Project`
- `Recent Projects` submenu in `File`
- reusable `Help` menu with:
  - `About`
  - `License`
  - `User Guide`

Recent UI work also improved layout, spacing, panel styling, notebook tabs, and button styling.

The app runs successfully with:

```bash
python CurveAnalyze.py
```

and the file passes:

```bash
python -m py_compile CurveAnalyze.py
```

## Architecture Overview
The app is still a single-file Tkinter application:

- `CurveAnalyze.py`
  - mathematical helper functions
  - `StyleManager`
  - `DataEntryTable`
  - `ELISAApplication`

Main architectural areas:

1. Math / fitting
- `four_param_logistic()`
- `three_param_logistic()`
- `calculate_r_squared()`
- fitting via `scipy.optimize.curve_fit`
- inversion via `scipy.optimize.brentq`

2. UI / layout
- `StyleManager` centralizes color and ttk styling
- `ELISAApplication` builds tabs, menus, dialogs, plotting area, and workflow actions
- `DataEntryTable` manages manual row-based standard entry

3. Project/session state
- session state is serialized to JSON for project save/load
- project state includes:
  - manual-entry table values
  - standard arrays
  - fit settings and fit results
  - single/bulk OD work
  - plot labels
- unsaved-change detection compares serialized current state against last saved/loaded state

4. Help/document dialogs
- `About` dialog is reusable from Help
- `License` and `User Guide` dialogs read from companion files beside the script
- file lookup is relative to `APP_DIR`, not `cwd`

5. Recent projects
- recent project paths persist in `recent_projects.json`
- capped at 5 entries
- most recent first

## Key Decisions
### Model behavior
- 3PL explicitly includes `D` in the formula, but fixes it at `0` using `THREE_PL_D = 0.0`
- 3PL supports increasing and decreasing data by choosing the initial slope sign and constraining `B` accordingly
- 4PL now allows both positive and negative `B`

### Standard-data constraints
- concentrations must be strictly greater than `0`
- this is intentional because the x-axis is plotted on a logarithmic scale
- a concentration of `0` is not allowed in the fitted standard set

### Help/document loading
- `LICENSE` and `README.md` are loaded relative to the script directory (`APP_DIR`)
- this fixes the old working-directory bug
- if only `CurveAnalyze.py` is copied to another folder without companion files, Help > License and Help > User Guide will fail by design

### Project persistence
- `Save` writes to the current project path if one exists
- `Save As...` always prompts for a new filename
- title bar changes to the saved/loaded project filename
- unsaved-change prompting is implemented before destructive session transitions

### UI choices
- recent UI refinement stayed inside Tkinter/ttk rather than introducing a new GUI framework
- buttons were softened within ttk limitations, but true modern rounded corners are not available in native ttk styling

## Next Steps
Highest-value remaining work:

1. Windows packaging preparation
- add `requirements.txt`
- make runtime resource handling robust for frozen builds if needed
- update icon loading to use `APP_DIR`
- prepare PyInstaller workflow
- prepare Inno Setup installer script

2. Windows validation
- test file dialogs
- test ttk rendering and fonts
- test right-click behavior
- test plot export and help dialogs in a packaged/frozen build

3. Optional workflow improvements
- export single-estimation results if desired
- add project metadata/version display consistency
- consider built-in fallback content for License/User Guide if companion files are absent
- consider prompting before other destructive operations if any are added later

## Notes / Constraints
- Project root currently contains:
  - `CurveAnalyze.py`
  - `LICENSE`
  - `README.md`
  - `__pycache__/`
- The app is still single-file and stateful; modifications should preserve current behavior unless explicitly changing workflow.
- The user has already tested most functional flows and considers the app functionally good at this stage.
- One deployment caveat remains intentional: `LICENSE` and `README.md` must stay beside the script unless a future embedded fallback is added.
- No Windows packaging artifacts have been added yet.
- Any future session should treat current functionality as the stable baseline and avoid regressing:
  - project save/load
  - unsaved-change prompts
  - recent-project list
  - 3PL/4PL trend handling
  - bulk CSV export
