default_cross_section_data_path = "./data/cross_section_data/"
default_barrett_moment_data_path = "./data/barrett_moment_data/"
default_spline_path = "./tmp/nucleus_splines/"
default_energy_path = "./tmp/binding_energies/"
default_fit_path = "./tmp/cross_section_fits/"

class paths():
    # 
    def __init__(self,cross_section_data_path,barrett_moment_data_path,spline_path,energy_path,fit_path):
        self.cross_section_data_path = cross_section_data_path
        self.barrett_moment_data_path = barrett_moment_data_path
        self.spline_path = spline_path
        self.fit_path = fit_path
        self.energy_path = energy_path
    #
    def print_paths(self):
        print('Imported Cross section data is saved at:',self.cross_section_data_path)
        print('Imported Barrett moments are saved at:',self.barrett_moment_data_path)
        print('Supporting points for splines are saved at:',self.spline_path)
        print('Binding energies are saved at:',self.energy_path)
        print('Fits are saved at:',self.fit_path)
    #
    def change_spline_path(self,path):
        self.spline_path = path
    #
    def change_fit_path(self,path):
        self.fit_path = path

local_paths=paths(default_cross_section_data_path,default_barrett_moment_data_path,default_spline_path,default_energy_path,default_fit_path)