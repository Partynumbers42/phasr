default_spline_path = "./tmp/splines/"
default_fit_path = "./tmp/fits/"

class paths():
    # 
    def __init__(self,spline_path,fit_path):
        self.spline_path = spline_path
        self.fit_path = fit_path
    #
    def print_paths(self):
        print('Splines are saved at:',self.spline_path)
        print('Fits are saved at:',self.fit_path)
    #
    def change_spline_path(self,path):
        self.spline_path = path
    #
    def change_fit_path(self,path):
        self.fit_path = path

local_paths=paths(default_spline_path,default_fit_path)