import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import SCHEMA_PATH
from tesseract_sql import DatabaseManager


@pytest.mark.parametrize("working_directory", [PROJECT_ROOT, PROJECT_ROOT / "src"])
def test_database_manager_resolves_schema_from_project_root(
    tmp_path, monkeypatch, working_directory
):
    database_path = tmp_path / "path-regression.sqlite"
    monkeypatch.chdir(working_directory)

    database = DatabaseManager(
        database_path,
        schema_path="resources/letters.sql",
    )

    try:
        assert database.schema_path == SCHEMA_PATH
        assert database.database_path == database_path.resolve()
        assert database.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='images'"
        ).fetchone() == ("images",)
    finally:
        database.conn.close()


def test_read_only_database_manager_rejects_missing_database(tmp_path):
    database_path = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError, match="Database does not exist"):
        DatabaseManager(database_path, create_if_missing=False)

    assert not database_path.exists()
