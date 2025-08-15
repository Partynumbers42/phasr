default_cross_section_data_path = "./data/cross_section_data/"
default_barrett_moment_data_path = "./data/barrett_moment_data/"
default_best_fit_path = "./data/best_fits/"
default_spline_path = "./tmp/nucleus_splines/"
default_energy_path = "./tmp/binding_energies/"
default_phase_shift_path = "./tmp/phase_shift/"
default_fit_path = "./tmp/cross_section_fits/"
default_correlation_quantities_path = "./tmp/correlation_quantities/"

class paths():
    # 
    def __init__(self,cross_section_data_path,barrett_moment_data_path,spline_path,energy_path,phase_shift_path,fit_path,best_fit_path,correlation_quantities_paths):
        self.cross_section_data_path = cross_section_data_path
        self.barrett_moment_data_path = barrett_moment_data_path
        self.spline_path = spline_path
        self.fit_path = fit_path
        self.best_fit_path = best_fit_path
        self.energy_path = energy_path
        self.phase_shift_path = phase_shift_path
        self.correlation_quantities_paths = correlation_quantities_paths
    #
    def print_paths(self):
        print('Imported Cross section data is saved at:',self.cross_section_data_path)
        print('Imported Barrett moments are saved at:',self.barrett_moment_data_path)
        print('Imported best fits are saved at:',self.best_fit_path)
        print('Supporting points for splines are saved at:',self.spline_path)
        print('Binding energies are saved at:',self.energy_path)
        print('Phase shifts are saved at:',self.phase_shift_path)
        print('Individual fits are saved at:',self.fit_path)
        print('Correlation quantities are saved at:',self.correlation_quantities_paths)
    #
    def change_spline_path(self,path):
        self.spline_path = path
    #
    def change_energy_path(self,path):
        self.energy_path = path
    #
    def change_fit_path(self,path):
        self.fit_path = path
    #
    def change_phase_shift_path(self,path):
        self.phase_shift_path = path
    #
    def change_correlation_quantities_path(self,path):
        self.correlation_quantities_paths = path

local_paths=paths(default_cross_section_data_path,default_barrett_moment_data_path,default_spline_path,default_energy_path,default_phase_shift_path,default_fit_path,default_best_fit_path,default_correlation_quantities_path)