



import yaml
import pandas as pd
from pathlib import Path

_dict = {'line_ID':[],'config':[],'Energy':[],'I_bbcw':350,'I_oct':[],'Nb':[],'emitt_strong':[],'emitt_weak':[],'xing': []}
for ff in sorted(list(Path('Tracking/').glob(f'coordinates_*.pkl'))):
    line_ID = ff.stem.split('coordinates_')[1]
    if 'emitt' in line_ID:
        continue

    
    _dict['line_ID'].append(line_ID)


    if 'xing' in line_ID:
        with open(f'Configs/config_{line_ID}.yaml') as fid:
            configuration = yaml.safe_load(fid)
        _dict['config'].append(f'Configs/config_{line_ID}.yaml')
        
        _dict['Energy'].append(configuration['beam_energy_tot'])
        _dict['Nb'].append(configuration['beam_npart'])
        _dict['I_oct'].append(configuration['oct_current'])
        
        assert float(configuration['beam_norm_emit_x'])== float(configuration['beam_norm_emit_y'])
        _dict['emitt_strong'].append(configuration['beam_norm_emit_x'])
        _dict['emitt_weak'].append(configuration['beam_norm_emit_x'])
        
        _dict['xing'].append(configuration['knob_settings']['on_x1'])
        continue
    if 'strong' in line_ID:
        with open(f'Configs/config_{line_ID.split("base_")[1]}.yaml') as fid:
            configuration = yaml.safe_load(fid)
        _dict['config'].append(f'Configs/config_{line_ID.split("base_")[1]}.yaml')
        
        _dict['Energy'].append(configuration['beam_energy_tot'])
        _dict['Nb'].append(configuration['beam_npart'])
        _dict['I_oct'].append(configuration['oct_current'])
        
        assert float(configuration['beam_norm_emit_x'])== float(configuration['beam_norm_emit_y'])
        _dict['emitt_strong'].append(configuration['beam_norm_emit_x'])
        _dict['emitt_weak'].append(2.3)
        
        _dict['xing'].append(configuration['knob_settings']['on_x1'])
        continue

    if ('base' in line_ID)&('strong' not in line_ID):
        with open(f'Configs/config_base.yaml') as fid:
            configuration = yaml.safe_load(fid)
        _dict['config'].append(f'Configs/config_base.yaml')
        
        _dict['Energy'].append(configuration['beam_energy_tot'])
        _dict['Nb'].append(configuration['beam_npart'])
        _dict['I_oct'].append(configuration['oct_current'])
        
        assert float(configuration['beam_norm_emit_x'])== float(configuration['beam_norm_emit_y'])
        _dict['emitt_strong'].append(configuration['beam_norm_emit_x'])
        
        if 'weak' in line_ID:
            emitt_ID = line_ID.split('base_weak_')[1]
            _dict['emitt_weak'].append({'01':1.9,'02':2.1,'03':2.3,'04':2.5}[emitt_ID])
        else:
            _dict['emitt_weak'].append(configuration['beam_norm_emit_x'])
        
        _dict['xing'].append(configuration['knob_settings']['on_x1'])

Allconfigs = pd.DataFrame(_dict).set_index('line_ID')


#%%

# IMPORTANT FUNCTIONS

import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sciStat
import pandas as pd
from IPython.display import clear_output
import sys
import rich
import pickle
import os


# os.chdir('/eos/user/p/phbelang/Programming/Octupoles_Guido/')

# source /home/phbelang/abp/BBStudies/Executables/py_BB/bin/activate
# if 'BBStudies/Executables/py_BB/bin' not in os.environ.get('PATH').split(':')[0]:
#     raise Exception('Wrong Python Distribution, use BBStudies/Executables/py_BB')



import xobjects as xo
import xtrack as xt
import xpart as xp
import xfields as xf

sys.path.append('/home/phbelang/abp/BBStudies/')
# Linux local
sys.path.append('/home/pbelanger/ABPlocal/BBStudies')
# Mac local
sys.path.append('/Users/pbelanger/ABPLocal/BBStudies/')
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.InteractionPoint as inp
import BBStudies.Physics.Detuning as dtune
import BBStudies.Physics.Base as phys
import BBStudies.Plotting.BBPlots as bbplt
import BBStudies.Physics.Constants as cst



def load_tracked(line_ID,sf=10,skipfirst=15,FOLDER = 'Tracking/'):
    tracked = {}
    for label in ['ref','HO','LR','OCTU','BBCW']:
        tracked[label] = pd.read_pickle(f'{FOLDER}/tracked_{label}_sf{sf}_{line_ID}.pkl')
    tracked['coord'] = pd.read_pickle(f'{FOLDER}/coordinates_{line_ID}.pkl')
    tracked['sf'] = sf
    _coord = tracked['coord']
    ctour_idx  = list(_coord[_coord['theta_sig'] == np.min(_coord['theta_sig'])].index)
    ctour_idx += list(_coord[_coord['r_sig'] == np.max(_coord['r_sig'])].index)[1:-1]
    ctour_idx += list(_coord[_coord['theta_sig'] == np.max(_coord['theta_sig'])].index)[::-1]
    ctour_idx += list(_coord[_coord['r_sig'] == np.min(_coord['r_sig'])].index)[1:-1][::-1]
    tracked['ctour_idx'] = ctour_idx[:-skipfirst]

    tracked['co'] = list(_coord[_coord['r_sig'] == np.min(_coord['r_sig'])].index)
    tracked['not_co'] = list(_coord[_coord['r_sig'] != np.min(_coord['r_sig'])].index)

    tracked['safe_theta'] = list(_coord[(_coord['theta_sig']>0.05*np.pi/2)&(_coord['theta_sig']<0.95*np.pi/2)].index)
    return tracked



def sum_footprints(_tracked,components,l_ID = 'base',contour = False,at_intensity = None,at_current = None,at_mo_current = None):
    dQx,dQy = np.zeros(len(_tracked['coord'])),np.zeros(len(_tracked['coord']))

    rescale = {'HO':1,'LR':1,'BBCW':1,'OCTU':1}
    if at_intensity is not None:
        rescale['HO'] = at_intensity/float(Allconfigs.loc[l_ID,'Nb'])
        rescale['LR'] = at_intensity/float(Allconfigs.loc[l_ID,'Nb'])
    if at_current is not None:
        rescale['BBCW'] = at_current/float(Allconfigs.loc[l_ID,'I_bbcw'])
    if at_mo_current is not None:
        rescale['OCTU'] = at_mo_current/float(Allconfigs.loc[l_ID,'I_oct'])

    for i in components:
    #---------------
        _dQx = _tracked[i].tunes_n['Qx'] - _tracked['ref'].tunes_n['Qx']
        _dQy = _tracked[i].tunes_n['Qy'] - _tracked['ref'].tunes_n['Qy']

        # Scale for scale_strength factor
        dQx += _dQx*_tracked['sf']*rescale[i]
        dQy += _dQy*_tracked['sf']*rescale[i]
    #---------------


    if contour:

        dQx,dQy = dQx.loc[_tracked['ctour_idx']],dQy.loc[_tracked['ctour_idx']]
    
    return dQx,dQy


def octupole_footprint(_tracked,_mo_df,gamma0,l_ID = 'base',contour=False):

    emitt = float(Allconfigs.loc[l_ID,'emitt_weak'])
    dQx,dQy = np.zeros(len(_tracked['coord'])),np.zeros(len(_tracked['coord']))

    for idx,mo in _mo_df.iterrows():
        #---------------
        _dQx,_dQy = dtune.DQx_DQy_octupole( _tracked['coord']['x_sig'],
                                            _tracked['coord']['y_sig'],
                                            betxy   = mo[['betx','bety']],
                                            emittxy = np.array([emitt*1e-6,emitt*1e-6])/gamma0,
                                            k1l     = 0,
                                            k3l     = mo['k3l'])


        dQx += _dQx
        dQy += _dQy
        #---------------

    if contour:
        dQx,dQy = dQx.loc[_tracked['ctour_idx']],dQy.loc[_tracked['ctour_idx']]
    
    return dQx,dQy


LINEFOLDER = './Lines/'
def import_from_lines(line_ID,to_track='b4'):
    line   = {}
    twiss  = {}
    survey = {}
    tracker_b1 = None
    tracker_b4 = None
    for seq in ['lhcb4']:
        _beam = seq[-2:]
        # Importing Line
        line[_beam] = xPlus.importLine(LINEFOLDER + f'line_{line_ID}_{seq}.json')
        
        # Importing twiss and tracker
        if _beam == 'b1':
            tracker_b1    = xt.Tracker(line=line[_beam])
            twiss[_beam]  = tracker_b1.twiss().to_pandas(index="name")
            survey[_beam] = tracker_b1.survey().to_pandas(index="name")
        elif _beam == 'b4':
            tracker_b4    = xt.Tracker(line=line[_beam])
            twiss[_beam]   = tracker_b4.twiss().to_pandas(index="name")
            survey[_beam]  = tracker_b4.survey().to_pandas(index="name")

            _beam = 'b2'
            twiss[_beam]   = tracker_b4.twiss().reverse().to_pandas(index="name")
            survey[_beam]  = tracker_b4.survey().reverse().to_pandas(index="name")


    if to_track == 'b1':
        tracker = tracker_b1
        del tracker_b4
    else:
        tracker = tracker_b4
        del tracker_b1
    # Clearing xsuite ouput
    clear_output(wait=False)



    allVars = list(tracker.vars._owner.keys())
    allElements = list(tracker.element_refs._owner.keys())


    # Deactivating all wires
    #===================================================
    tracker.vars['enable_qff'] = 0
    for IP in ['ip1','ip5']:
        tracker.vars[f"bbcw_rw_{IP}.{_beam}"] = 1
        tracker.vars[f"bbcw_i_{IP}.{_beam}"]  = 0
    run3_wires = [name for name in allElements if ('bbcw' in name)&('wire' in name)]
    for wire in run3_wires:
        tracker.line.element_dict[wire].post_subtract_px  = 0
        tracker.line.element_dict[wire].post_subtract_py  = 0
        
    # Creating BB knobs
    #===================================================
    
    for _ip in ['ip1','ip5','ip2','ip8']:
        bb_lr = [name for name in allElements if ('bb_lr' in name)&(f'{_ip[-1]}{_beam}' in name)]
        bb_ho = [name for name in allElements if ('bb_ho' in name)&(f'{_ip[-1]}{_beam}' in name)]

        # New knob:
        tracker.vars[f'{_ip}_bblr_ON'] = 1
        tracker.vars[f'{_ip}_bbho_ON'] = 1

        # Linking to new knob 
        for _lr in bb_lr:
            # Infividual knobs
            tracker.vars[f'{_lr}_ON'] = 1
            tracker.element_refs[_lr].scale_strength = tracker.vars[f'{_lr}_ON']*tracker.vars[f'{_ip}_bblr_ON']*tracker.element_refs[_lr].scale_strength._value

        for _ho in bb_ho:
            tracker.element_refs[_ho].scale_strength = tracker.vars[f'{_ip}_bbho_ON']*tracker.element_refs[_ho].scale_strength._value

    for _ip in ['ip1','ip5','ip2','ip8']:
        tracker.vars[f'{_ip}_bblr_ON'] = 0
        tracker.vars[f'{_ip}_bbho_ON'] = 0


    # Creating sext and oct knobs
    #====================================================

    # AS DONE IN THE MASK
    #---------------------
    part = tracker.line.particle_ref
    brho = part.p0c[0]/(part.q0*cst.c)
    tracker.vars['I_oct'] = 0
    for ss in '12 23 34 45 56 67 78 81'.split():
        tracker.vars[f'kof.a{ss}{_beam}'] = tracker.vars['kmax_mo']*tracker.vars['I_oct']/tracker.vars['imax_mo']/brho
        tracker.vars[f'kod.a{ss}{_beam}'] = tracker.vars['kmax_mo']*tracker.vars['I_oct']/tracker.vars['imax_mo']/brho
    #---------------------

    
    ks = [name for name in allVars if ('ksf' in name)|('ksd' in name)]
    ko = [name for name in allVars if ('kof.a' in name)|('kod.a' in name)]

    tracker.vars['all_oct_ON']  = 1
    tracker.vars['all_sext_ON'] = 1
    for _ks in ks:
        if tracker.vars[_ks]._expr is None:
            tracker.vars[_ks] = tracker.vars['all_sext_ON']*tracker.vars[_ks]._value
        else:
            tracker.vars[_ks] = tracker.vars['all_sext_ON']*tracker.vars[_ks]._expr 
    for _ko in ko:
        if tracker.vars[_ko]._expr is None:
            tracker.vars[_ko] = tracker.vars['all_oct_ON']*tracker.vars[_ko]._value
        else:
            tracker.vars[_ko] = tracker.vars['all_oct_ON']*tracker.vars[_ko]._expr 

    return tracker,line,twiss,survey



