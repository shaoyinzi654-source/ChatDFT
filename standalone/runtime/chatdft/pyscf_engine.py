"""Open source quantum calculation process. Runs on Linux; no user code execution."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys
import time


METHODS={'B3LYP':'b3lyp','PBE1PBE':'pbe0','M062X':'m06-2x','CAM-B3LYP':'cam-b3lyp','HF':'HF','MP2':'MP2'}
BASES=['6-31G(d)','6-31+G(d,p)','6-311+G(d,p)','Def2SVP','Def2TZVP','cc-pVDZ','STO-3G']
EPS={'Water':78.3553,'Ethanol':24.852,'Methanol':32.613,'Acetonitrile':35.688,'DMSO':46.826,'Toluene':2.3741,'Chloroform':4.7113,'THF':7.4257}


def run(input_path,output_path):
    import numpy as np
    from pyscf import gto,dft,scf,mp,lib,tdscf
    from pyscf.hessian import thermo
    value=json.loads(Path(input_path).read_text('utf-8'))
    s,c=value['structure'],value['calculation']
    if c['method'] not in METHODS or c['basis'] not in BASES or c['task'] not in ['sp','opt','optfreq','freq','td']:
        raise ValueError('开源引擎尚未支持此方法或任务。')
    if c['solvent']!='Gas' and c['solvent'] not in EPS:raise ValueError('未知溶剂。')
    if c.get('dispersion'):raise ValueError('当前 PySCF 适配器未启用附加 D3BJ；可使用自带色散参数的方法，或关闭附加校正。')
    if c['method']=='MP2' and (c['task']!='sp' or c['solvent']!='Gas'):raise ValueError('当前 MP2 适配器仅支持气相单点。')
    if not 1<=int(c['cores'])<=16 or not 1<=int(c['memory_gb'])<=14:raise ValueError('资源配置超限。')
    if not 1<=len(s['atoms'])<=500:raise ValueError('原子数量超限。')
    for a in s['atoms']:
        if not all(math.isfinite(float(a[k])) for k in ['x','y','z']):raise ValueError('坐标无效。')
    effective_cores = min(int(c['cores']), int(os.environ.get('CHATDFT_EFFECTIVE_CORES', c['cores'])))
    lib.num_threads(effective_cores)
    geometry=[(a['element'],[a[k] for k in ['x','y','z']]) for a in s['atoms']]
    mol=gto.M(atom=geometry,basis=c['basis'],charge=int(s['charge']),spin=int(s['multiplicity'])-1,unit='Angstrom',verbose=4,max_memory=int(c['memory_gb'])*1024)
    energies=[]
    def make_mf(m):
        if c['method'] in ['HF','MP2']:
            mf=scf.RHF(m) if m.spin==0 else scf.UHF(m)
        else:
            mf=dft.RKS(m) if m.spin==0 else dft.UKS(m)
            mf.xc=METHODS[c['method']];mf.grids.level=3
        mf.conv_tol=1e-9;mf.max_cycle=120;mf.chkfile='calculation.chk'
        if c['solvent']!='Gas':
            mf=mf.PCM();mf.with_solvent.method='IEF-PCM';mf.with_solvent.eps=EPS[c['solvent']]
        def callback(env):
            if 'e_tot' in env:energies.append(float(env['e_tot']))
        mf.callback=callback
        return mf
    mf=make_mf(mol);mf.kernel()
    if not mf.converged:raise RuntimeError('SCF 未收敛；请修改初始结构、方法或自旋后重试。')
    optimized=False
    if c['task'] in ['opt','optfreq']:
        from pyscf.geomopt.geometric_solver import kernel
        optimized,mol=kernel(mf,maxsteps=100,assert_convergence=True)
        if not optimized:raise RuntimeError('结构优化未达到收敛标准。')
        mf=make_mf(mol);mf.kernel()
        if not mf.converged:raise RuntimeError('优化后的 SCF 未收敛。')
    energy=float(mf.e_tot)
    if c['method']=='MP2':
        correlation=mp.MP2(mf);correlation.kernel();energy=float(correlation.e_tot)
    result=dict(engine='PySCF',engine_version=__import__('pyscf').__version__,normal_termination=True,
                scf_converged=bool(mf.converged),optimization_converged=bool(optimized),
                final_energy_hartree=energy,scf_energies_hartree=energies,
                frequencies_cm1=[],ir_intensities_km_mol=[],imaginary_frequencies=[],warnings=[],excited_states=[],
                zero_point_hartree=None,gibbs_hartree=None,enthalpy_hartree=None,
                nmr_shieldings_ppm=[],charge=mol.charge,multiplicity=mol.spin+1,
                calculation=c,effective_cores=effective_cores,solvent_model='Gas' if c['solvent']=='Gas' else 'IEF-PCM',
                geometry=[dict(element=mol.atom_symbol(i),x=float(p[0]),y=float(p[1]),z=float(p[2])) for i,p in enumerate(mol.atom_coords(unit='Angstrom'))])
    result['dipole_debye']=float(np.linalg.norm(mf.dip_moment(unit='Debye',verbose=0)))
    _,charges=mf.mulliken_pop(verbose=0)
    result['mulliken_charges']=[dict(index=i+1,element=mol.atom_symbol(i),charge=float(x)) for i,x in enumerate(charges)]
    for spin,index in [('alpha',0),('beta',1)]:
        if mol.spin==0:
            e=np.asarray(mf.mo_energy);occ=np.asarray(mf.mo_occ)>0
        else:
            e=np.asarray(mf.mo_energy[index]);occ=np.asarray(mf.mo_occ[index])>0
        homo=float(max(e[occ]))*27.211386245988 if np.any(occ) else None
        lumo=float(min(e[~occ]))*27.211386245988 if np.any(~occ) else None
        result[spin+'_homo_ev']=homo;result[spin+'_lumo_ev']=lumo;result[spin+'_gap_ev']=lumo-homo if homo is not None and lumo is not None else None
    if c['task'] in ['freq','optfreq']:
        h=mf.Hessian().kernel()
        vibration=thermo.harmonic_analysis(mol,h)
        raw=np.asarray(vibration['freq_wavenumber'])
        freq=[-abs(float(x.imag)) if abs(x.imag)>1e-8 else float(x.real) for x in raw.astype(complex)]
        result['frequencies_cm1']=freq;result['imaginary_frequencies']=[x for x in freq if x < -20]
        if result['imaginary_frequencies']:result['warnings'].append('存在显著虚频；请检查对应模式与驻点性质。')
        if any(-20<=x<0 for x in freq):result['warnings'].append('存在小负频率，需要检查软模式和数值精度。')
        if not any(x<0 for x in freq):
            thermal=thermo.thermo(mf,vibration['freq_au'],298.15,101325)
            for key,target in [('ZPE','zero_point_hartree'),('G_tot','gibbs_hartree'),('H_tot','enthalpy_hartree')]:
                if key in thermal:result[target]=float(thermal[key][0])
            result['thermochemistry']='Ideal gas RRHO, 298.15 K, 1 atm; unscaled harmonic frequencies'
            if c['solvent']!='Gas':result['warnings'].append('溶剂中使用气相 RRHO 热修正；未执行溶液标准态修正。')
        result['warnings'].append('当前适配器输出谐振频率；尚未计算 IR 强度，不生成伪造 IR 光谱。')
    if c['task']=='td':
        td=mf.TDHF() if c['method']=='HF' else mf.TDDFT()
        td.nstates=min(10,max(1,mol.nao_nr()-mol.nelectron//2));td.kernel()
        if not np.all(td.converged):raise RuntimeError('部分激发态未收敛。')
        strengths=td.oscillator_strength()
        for i,(e,f) in enumerate(zip(td.e,strengths)):
            ev=float(e)*27.211386245988
            result['excited_states'].append(dict(state=i+1,label='TDHF' if c['method']=='HF' else 'TDDFT',energy_ev=ev,wavelength_nm=1239.841984/ev,oscillator_strength=float(f)))
    if mol.spin:
        result['spin_square']=float(mf.spin_square()[0]);result['warnings'].append('开壳层使用 UHF/UKS；请检查 <S²> 与自旋污染。')
    result['completed_at']=time.time()
    tmp=Path(str(output_path)+'.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,output_path)
    print('CHATDFT PYSCF COMPLETED',flush=True)


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2])
