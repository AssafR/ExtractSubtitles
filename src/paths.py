from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "resources" / "letters.sql"
CANONICAL_DATABASE_PATH = PROJECT_ROOT / "letters.sqlite"
DEFAULT_DATABASE_PATH = CANONICAL_DATABASE_PATH


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()
