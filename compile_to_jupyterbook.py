import os
import shutil

root_src_dir = './source_code/'
root_dst_dir = './juliabook/'

shutil.copytree(root_src_dir, root_dst_dir, dirs_exist_ok=True)

import script_to_myst_Julia as jlscript

def remove_files(root_src_dir, root_dst_dir):
    for file in os.listdir(root_src_dir):
        if os.path.isfile(os.path.join(root_src_dir, file)):
            os.remove(os.path.join(root_dst_dir, file))
        else:
            shutil.rmtree(os.path.join(root_dst_dir, file))

jlscript.create_myst_nb()

#if jlscript.create_myst_nb():
os.system("jupyter-book build juliabook/")
remove_files(root_src_dir, root_dst_dir)
    #print("Success!")
#exit(0)
#else:
    #remove_files(root_src_dir, root_dst_dir)
    #print("Failed! - Directory cleaned")
    #exit(1)
