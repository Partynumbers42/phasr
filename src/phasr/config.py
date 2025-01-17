default_spline_path = "./tmp/nucleus_splines/"
default_energy_path = "./tmp/binding_energies/"
default_fit_path = "./tmp/crosssection_fits/"

class paths():
    # 
    def __init__(self,spline_path,energy_path,fit_path):
        self.spline_path = spline_path
        self.fit_path = fit_path
        self.energy_path = energy_path
    #
    def print_paths(self):
        print('Splines are saved at:',self.spline_path)
        print('Binding energies are saved at:',self.energy_path)
        print('Fits are saved at:',self.fit_path)
    #
    def change_spline_path(self,path):
        self.spline_path = path
    #
    def change_fit_path(self,path):
        self.fit_path = path

local_paths=paths(default_spline_path,default_energy_path,default_fit_path)