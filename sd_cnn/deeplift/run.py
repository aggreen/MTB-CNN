import os
drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']
for drug in drugs:
    print(f"sbatch run.sh {drug}")
    os.system(f"sbatch run.sh {drug}")
