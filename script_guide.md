# How does one use the scripts?

1. Install the dependencies
    - `pip install -r requirements.txt` for all the Python packages.
    - Instantiating using ./julia_version/Project.toml for the Julia packages.
2. Update the source code to your needs.
3. Tweak file structures and function calls to your needs in the `chapter_meta` dictionary in either script.
4. Run the script.
    - `python3 create_juliabook.py` for the Julia version.
    - `python3 create_pythonbook.py` for the Python version.
5. The generated books and associated figures can be found in the `./julia_version/` and `./python_version/` directories respectively.