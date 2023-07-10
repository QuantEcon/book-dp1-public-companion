import os
import shutil

root_src_dir = './source_code/'
root_dst_dir = './juliabook/'

shutil.copytree(root_src_dir, root_dst_dir, dirs_exist_ok=True)

import juliabook.script_to_myst_Julia as jlscript

jlscript.create_myst_nb()

os.system("jupyter-book build juliabook/")

