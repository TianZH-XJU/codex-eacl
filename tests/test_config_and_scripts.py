from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_required_project_docs_exist():
 assert (ROOT/"AGENTS.md").exists()
 assert (ROOT/"docs"/"PROJECT_PLAN.md").exists()
 assert (ROOT/"STATUS.md").exists()
def test_pinned_dataset_revisions_are_present():
 script=(ROOT/"scripts"/"download_datasets.py").read_text()
 assert "18072d2685ea682290f7b8924d94c62acc19c0b2" in script
 assert "155b9c710419136e17307b80d0a13e68cd46b4ec" in script
 assert "f7672a018e8ceb37fc0184dcfbb7e665155ffea6" in script
def test_cpu_device_is_explicit_in_smoke_runner():
 script=(ROOT/"scripts"/"run_baseline_smoke.py").read_text()
 assert '"cpu"' in script
 assert '"--device", "cpu"' in script
