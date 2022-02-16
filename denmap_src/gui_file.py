from tkinter import filedialog, messagebox, END, DISABLED
from PIL import Image
from os import path
import cv2
import numpy as np

def gui_load_image(self):
    self.img_name = filedialog.askopenfilename(title = "Choose an image to load",filetypes = [('All image file types', '.bmp .dib .jpeg .jpg .jpe .jp2 .png .pbm .pgm .ppm .sr .ras .tiff .tif'), ('Windows bitmaps', '.bmp .dib'),('JPEG', '.jpeg .jpg .jpe .jp2'), ('Portable Network Graphics','.png'),('Portable image format','.pbm .pgm .ppm'),('Sun rasters','.sr .ras'),('Tagged Image File Format','.tiff .tif')])
    if (self.img_name):
        self.main_image = cv2.imread(self.img_name,0)
        self.original_image = self.main_image
        self.gui_update_image(Image.fromarray(self.main_image))
        self.window.title(f"DenMap \u00A9 {self.version_text} - {path.basename(self.img_name)} - {self.main_image.shape[1]}x{self.main_image.shape[0]}")
        self.p_centres = False
        self.circle_image = False
        self.new_image = False
        self.binary_image = False
        self.NCC_values = False
        self.NCC_temp = False
        self.NCC_thres = False
        self.analysis_data = False
        self.load_state = 0
        self.gui_disable_menu()
        self.extra_pts = []
        self.zoom_scale = 1
        self.zoom_corner = [0,0]
        self.FFT_in = False
        self.FFT_out = 100
        self.pixel_scaling = [1,1,1]
        self.distance_unit = "pixels"
        self.highlight_pts = []
        self.unset_view_variables()
        self.menu_analyse.entryconfig(0, state="normal")
        self.menu_view.entryconfig(0, state="normal")
        self.image_checkvariables[0].set(1)
        self.menu_image.entryconfig(0,state="normal")
        self.menu_process.entryconfig(0,state="normal")
        self.menu_process.entryconfig(1,state="normal")
        self.menu_process.entryconfig(2,state="normal")
        self.menu_file.entryconfig(1,state="normal")

def gui_update_title(self):
    title = f"DenMap \u00A9 {self.version_text} - {path.basename(self.img_name)} - {self.main_image.shape[1]}x{self.main_image.shape[0]}"
    if self.distance_unit != "pixels":
        sf_pos = np.log10(abs(self.pixel_scaling[2]))
        if sf_pos < -1:
            title += f" - {self.pixel_scaling[2]/10**(int(sf_pos)-1):.2f}e{int(sf_pos)-1} {self.distance_unit}/pixel"
        elif sf_pos > 3:
            title += f" - {self.pixel_scaling[2]/10**(int(sf_pos)):.2f}e{int(sf_pos)} {self.distance_unit}/pixel"
        else:
            title += f" - {self.pixel_scaling[2]:.2f} {self.distance_unit}/pixel"
    self.window.title(title)

def gui_load_points(self):
    if type(self.main_image) == bool:
        messagebox.showerror("No image loaded!","Points cannot be plotted without an image.")
        return
    data_name = filedialog.askopenfilename(title = "Choose points file to load",filetypes = [('Comma Separated Value file','.csv')])
    if not path.isfile(data_name):
        return
    csv_title = ""
    data = []
    with open(data_name,"r") as f:
        csv_title = f.readline().split(",")
        for line in f.readlines():
            data += [line.split(",")]
    data = np.array(data,dtype=float)
    if len(data[0]) != 2 and len(data[0]) != 3 and len(data[0]) != 5:
        messagebox.showerror("Incorrect data file!","Please load a valid points file.")
        return
    if len(data) < 10:
        messagebox.showerror("Insufficient number of points","Please load at least 10 points.")
        return
    self.highlight_pts = []
    if len(data[0]) == 3 or len(data[0]) == 5:
        self.p_centres = np.round(data[data[:,-1] == 0,:2]).astype(int)
        self.extra_pts = list(np.round(data[data[:,-1] == 1,:2]).astype(int))
    else:
        self.p_centres = np.round(data[:,:2]).astype(int)

    if len(data[0]) == 5:
        #Calculate scaling
        global distance_unit, pixel_scaling
        self.distance_unit = csv_title[3][csv_title[3].find("(")+1:-1].replace('u',u'\u03BC')
        pts = np.array([data[:,0],data[:,1]]).flatten()
        dist_points = np.array([data[:,2],data[:,3]]).flatten()
        calc_scaling = np.mean(dist_points/pts)
        self.pixel_scaling = [1/calc_scaling,1,calc_scaling]
        self.gui_update_title()
    self.load_state = 1
    self.main_image = self.gui_draw_circles(self.original_image,self.p_centres,return_np=True)
    self.circle_image = Image.fromarray(self.main_image)
    self.gui_update_image(self.circle_image)
    last = self.menu_analyse.index(END)+1
    for i in range(1,last):
        self.menu_analyse.entryconfig(i, state="normal")
    self.menu_export.entryconfig(1,state="normal")
    self.menu_export.entryconfig(2,state="normal")
    self.menu_export.entryconfig(3,state="normal")
    self.menu_view.entryconfig(3,state="normal",label='Image with Centres')
    self.menu_process.entryconfig(0,state=DISABLED)
    self.menu_process.entryconfig(1,state=DISABLED)
    self.menu_process.entryconfig(2,state=DISABLED)
    self.image_checkvariables[0].set(0)
    self.image_checkvariables[3].set(1)

def gui_quit(self):
    self.window.destroy()