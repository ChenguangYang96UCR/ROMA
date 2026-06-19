# Baseline Code Setup

This folder stores the upstream implementations used as baseline models in our
experiments. Each baseline is kept in a separate subdirectory so that the
original source code and our project-specific adaptation scripts remain easy to
track.

## Folder Structure

```text
baseline/
├── BigST/
├── LargeST/
├── PatchSTG/
└── ST-WA/
```

## Clone Baseline Repositories

Run the following commands from the project root:

```bash
mkdir -p baseline
cd baseline

git clone git@github.com:usail-hkust/BigST.git BigST
git clone git@github.com:liuxu77/LargeST.git LargeST
git clone git@github.com:razvanc92/ST-WA.git ST-WA
git clone git@github.com:LMissher/PatchSTG.git PatchSTG
```

If SSH access is not configured, use HTTPS instead:

```bash
mkdir -p baseline
cd baseline

git clone https://github.com/usail-hkust/BigST.git BigST
git clone https://github.com/liuxu77/LargeST.git LargeST
git clone https://github.com/razvanc92/ST-WA.git ST-WA
git clone https://github.com/LMissher/PatchSTG.git PatchSTG
```

## Source Repositories

| Folder | Baseline | Upstream repository |
|---|---|---|
| `BigST/` | BigST | `https://github.com/usail-hkust/BigST` |
| `LargeST/` | LargeST baseline framework, including models such as STGCN, GWN, and ASTGCN | `https://github.com/liuxu77/LargeST` |
| `PatchSTG/` | PatchSTG | `https://github.com/LMissher/PatchSTG.git` |
| `ST-WA/` | ST-WA | `https://github.com/razvanc92/ST-WA` |

## Record Repository Versions

After cloning the repositories, record the exact commit hash used in the
experiments:

```bash
for repo in BigST LargeST PatchSTG ST-WA; do
  echo "===== ${repo} ====="
  git -C "${repo}" rev-parse HEAD
done
```

This makes the baseline setup reproducible and avoids ambiguity if upstream
repositories change later.

## Notes

- Keep upstream baseline code in the corresponding subdirectory.
- Keep project-specific conversion, adaptation, and run scripts separate from
  the original code when possible.
- If a baseline requires local modifications for our Philadelphia service
  request datasets, document the modified files and commands in that baseline's
  subdirectory.
