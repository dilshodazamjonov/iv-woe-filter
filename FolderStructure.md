```text
IvPackage/
|-- .github/
|   `-- workflows/
|       `-- ci.yaml
|-- src/
|   |-- iv_woe_filter/
|   |   |-- __init__.py
|   |   |-- binning.py
|   |   |-- iv_woe_filter.py
|   |   |-- metrics.py
|   |   |-- plots.py
|   |   |-- validation.py
|   |   `-- woe.py
|   `-- iv_woe_filter.egg-info/
|       |-- dependency_links.txt
|       |-- PKG-INFO
|       |-- requires.txt
|       |-- SOURCES.txt
|       `-- top_level.txt
|-- tests/
|   |-- conftest.py
|   |-- test_artifacts_and_psi.py
|   |-- test_binning_behavior.py
|   |-- test_fit_transform.py
|   |-- test_metrics.py
|   |-- test_single_class_target.py
|   |-- test_tree_binning.py
|   `-- test_validation_and_selection.py
|-- audit/
|-- dist/
|-- smoke-sdist/
|-- smoke-wheel/
|-- .gitignore
|-- .python-version
|-- FolderStructure.md
|-- LICENSE
|-- main.py
|-- pyproject.toml
|-- README.md
`-- uv.lock
```

Notes:

- `src/iv_woe_filter/` contains the actual package source.
- `tests/` contains the current pytest suite.