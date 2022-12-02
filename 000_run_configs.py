import subprocess
import sys
import os
from pathlib import Path


# source /home/phbelang/abp/BBStudies/Executables/py_BB/bin/activate
if 'BBStudies/Executables/py_BB/bin' not in os.environ.get('PATH').split(':')[0]:
    raise Exception('Wrong Python Distribution, use BBStudies/Executables/py_BB')

# NOTE: Make sure you add: 
# pm.install_lenses_in_sequence(mad_track, bb_dfs['b2'], 'lhcb2')
# at line 438 of '000_pymask_rich.py'{}
# Saving sequences and BB dfs



# Running pymask
#===========================
CONFIG_PATH  = '/eos/user/p/phbelang/Programming/FPCatalogue/Configs'
OUTPUTFOLDER = f'/eos/user/p/phbelang/Programming/FPCatalogue/Lines'
#===========================

for config_file in list(Path(CONFIG_PATH).glob(f'config_*.yaml')):
    for mode in ['b1_with_bb','b4_from_b2_with_bb']:
        
        seq = {'b1_with_bb': 'lhcb1', 'b4_from_b2_with_bb' : 'lhcb4'}[mode]

        line_file = f"line_{config_file.stem.split('config_')[1]}_{seq}.json"

        cwd = os.getcwd()
        os.chdir('/home/phbelang/abp/BBStudies/Data/Mask/')
        template = open("000_mask_template_rich.py").read()
    
        
        # CHOOSING GOOD CONFIG:
        template = template.replace("open('config.yaml','r')",f"open('{str(config_file)}','r')")

        # CHOOSING GOOD MODE
        template = template.replace("mode = configuration['mode']",f"mode = '{mode}'")

        # Exporting lines correctly:
        template = template.replace("folder_name = './xsuite_lines'",f"folder_name = '{OUTPUTFOLDER}', file_name = '{line_file}'")
        
        
        
        exec(template)
        
        os.chdir(cwd)

        

