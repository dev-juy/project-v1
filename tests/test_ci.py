"""Guard the CI contract: the workflow must run the full suite verbosely."""

import os

WORKFLOW = os.path.join(os.path.dirname(__file__), "..",
                        ".github", "workflows", "tests.yml")


def test_ci_workflow_runs_full_suite():
    assert os.path.exists(WORKFLOW), "GitHub Actions workflow missing"
    with open(WORKFLOW) as f:
        content = f.read()
    assert "python -m pytest tests/ -v" in content
    assert "pip install -r requirements.txt" in content
