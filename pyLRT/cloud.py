import os
import subprocess
import tempfile
import numpy as np
import io
import xarray as xr

class Cloud:
    def __init__(self, lrt_folder, cwc, eff_radius, habit=None):
        self.folder = lrt_folder
        self.cwc = cwc # Cloud water content (g/m3)
        self.eff_radius = eff_radius # Effective radius (um)
        self.habit = habit # Cloud ice crystal habit, from:          
        #  0 : Solid column
        #  1 : Hollow column
        #  2 : Rough aggregate
        #  3 : Rosette 4
        #  4 : Rosette 6
        #  5 : Plate
    
    def cldprp(self, wvl, method="k"):
        tmpcloud = tempfile.NamedTemporaryFile(delete=False)
        cloudstr = '{:4.2f} {:4.2f} {:4.2f} '.format(
                wvl,
                self.cwc,
                self.eff_radius)
        if self.habit is not None:
            cloudstr += '{} '.format(self.habit)
        cloudstr += '\n'
        tmpcloud.write(cloudstr.encode())
        tmpcloud.close()
        
        cwd = os.getcwd()
        os.chdir(os.path.join(self.folder, 'bin'))

        process = subprocess.run([os.getcwd()+f'/cldprp', f'-k', f'{tmpcloud.name}'], 
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 encoding='ascii')
        os.remove(tmpcloud.name)
        os.chdir(cwd)
        
        err = io.StringIO(process.stderr)
        for line in err.readlines():
            if 'warning' in line:
                print(line.strip())
        
        out_cols = [
            'wvl',
            'cwc',
            'r_eff',
            'ext',
            'asy',
            'ssa',
            'g1',
            'g2',
            'f'
        ]
        data = np.genfromtxt(io.StringIO(process.stdout))
        return xr.Dataset(dict(zip(out_cols[:len(data)], data)))

if __name__ == "__main__":
    cld = Cloud('/Users/ogd22/libRadtran-2.0.5', 0.1, 10)
    print(cld.cldprp(550))