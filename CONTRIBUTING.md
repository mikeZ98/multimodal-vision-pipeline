# Contributing to multimodal-vision-pipeline

## ⛔ Sole-Contributor Invariant (non-negotiable)

Every commit in this repository, on every branch, MUST have **author and committer** set to exactly:

| Field | Value |
|-------|-------|
| Name  | `mikeZ98` |
| Email | `145975972+mikeZ98@users.noreply.github.com` |

- No other name, no machine username, no personal or local email: zero tolerance.
- Commit messages follow Conventional Commits. AI-native development is welcome and disclosed:
  tool-assistance trailers such as `Co-Authored-By:` may stay in the message, and every change
  passes the same automated gates.
- Every GitHub web-UI merge method (merge, squash and rebase) records `GitHub <noreply@github.com>`
  as committer and **will fail** the audit. Fast-forward `main` locally instead.

### How it is enforced

| Layer | Where | Checks |
|-------|-------|--------|
| `pre-commit` | `.githooks/pre-commit` | author/committer identity; blocks `.cache/`, venvs, `.env*`, keys; ruff check and format on staged Python |
| `commit-msg` | `.githooks/commit-msg` | Conventional Commit subject; identity re-check (covers `--author` and `--amend`) |
| `pre-push` | `.githooks/pre-push` | full audit of every commit being pushed |
| Manual / CI | `scripts/audit-attribution.sh` | configured identity plus a full audit of the branch history |

The policy (allowed identity and Conventional Commit format) lives in one place: `.githooks/_policy.sh`.
Audit any range by hand with `scripts/audit-attribution.sh [<rev-list args>]` (default: `HEAD`).

Never bypass hooks with `--no-verify`.

## Environment setup (macOS)

All caches, the virtualenv, temp files and the Colima VM live under `./.cache/` on
the project root, so nothing is written outside the project.

```bash
brew install uv direnv poppler
direnv allow          # pins caches and sets core.hooksPath + the local git identity
uv sync               # downloads CPython 3.12 and creates .cache/venv
```

`poppler` provides the `pdftoppm` binary that `pdf2image` drives at runtime. Tests never call it:
rasterization is faked in-memory so the suite stays hermetic.

## Quality gates

```bash
ruff check . && ruff format --check .
mypy                      # strict, configured in pyproject.toml
bandit -c pyproject.toml -r src
pytest                    # hermetic: sockets disabled, coverage ≥ 90% (branch)
scripts/audit-attribution.sh
```

## Commit discipline

- [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <summary>`, subject
  ≤ 100 characters. Types: `feat fix chore docs test build ci refactor perf style revert`.
- One logical change per commit. Keep commits atomic and green.
- Dependency updates: `uv lock --upgrade-package <name>`, run the gates, commit `uv.lock`. Bot-authored
  update commits (e.g. Dependabot) fail the audit by design.
