# FACTORY PROTOCOLS

## 0. Context & Scope (CRITICAL)
- **Project Type:** Machine Learning Framework (C#).
- **Target Directories:**
  - Source Code: `src/` ONLY.
  - Unit Tests: `tests/` ONLY.
- **FORBIDDEN FILES:**
  - `orchestrator.py` (SYSTEM FILE - DO NOT TOUCH)
  - `opencode.json` (SYSTEM FILE - DO NOT TOUCH)
  - `factory_manager.py` (SYSTEM FILE - DO NOT TOUCH)

## 1. The Prime Directive
- **Designer** writes to `0_ideas/` ONLY. Focus on ML architectures, optimizers, and data loaders.
- **Architect** reads `0_ideas/` -> writes `1_specs/`. Focus on class hierarchies and API design.
- **Coder** reads `1_specs/` -> writes `src/` & `tests/`.

## 2. File Handoffs
- NEVER work on a file unless it is in your designated input folder.
- Move files to the next folder when done (e.g., `mv 1_specs/task.md 2_in_progress/task.md`).

## 3. Git Hygiene (STRICT)
All code changes must be committed to the `ai` branch.
**Commit Steps:**
1. `git status`
2. `git add .`
3. `git commit -m "[CODER] <Description>"`
4. `git push fork ai`