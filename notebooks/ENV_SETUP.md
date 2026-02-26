# Installing packages into the notebook kernel

## The rule

**The same Python that runs the notebook kernel must be the one that gets `pip install`.**  
If you install from a different terminal/env, the notebook may still see the old (or no) packages.

## How to know you're installing into the correct environment

### 1. See which Python the kernel uses

In any notebook, run:

```python
import sys
print(sys.executable)
```

Example output:  
`C:\Users\...\AppData\Local\Programs\Python\Python311\python.exe`  
or  
`C:\Users\...\miniconda3\envs\myenv\python.exe`

That path **is** the correct environment for this kernel.

### 2. Use that Python for pip (any terminal)

In **any** terminal (PowerShell, CMD, or Bash), run:

```bash
"<path from step 1>" -m pip install -r notebooks/requirements-notebooks.txt
```

Examples:

- **PowerShell:**  
  `& "C:\Users\...\Python\Python311\python.exe" -m pip install -r notebooks/requirements-notebooks.txt`
- **Bash (Git Bash / WSL):**  
  `"/c/Users/.../Python/Python311/python.exe" -m pip install -r notebooks/requirements-notebooks.txt`

Using **`-m pip`** ensures you're using the pip that belongs to that Python, so you're always installing into the correct environment. The terminal type (Bash vs PowerShell) doesn't change that—only which executable you call does.

### 3. Optional: install from inside the notebook

To avoid dealing with paths, you can install from a notebook cell (uses the kernel's Python automatically):

```python
!pip install -r requirements-notebooks.txt
```

Run this from the `notebooks/` folder, or use the full path to the requirements file.

## If you use a virtual environment (venv / conda)

- **venv:** Activate it in your terminal (`.\venv\Scripts\activate` on Windows, `source venv/bin/activate` on Bash), then run `python -m pip install -r notebooks/requirements-notebooks.txt`. In Cursor/VS Code, choose **Kernel → Select Kernel** and pick the interpreter from that venv (e.g. `./venv/Scripts/python.exe`).
- **conda:** Activate the env (`conda activate myenv`), then `python -m pip install -r notebooks/requirements-notebooks.txt`. In the notebook, select the kernel that shows that conda env.

After that, the “which Python” cell above will show the venv/conda Python path; using that path with `-m pip` in any terminal still installs into the correct environment.
