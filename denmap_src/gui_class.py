from PIL import Image, ImageTk
from tkinter import Tk, Menu, Canvas, BooleanVar, DISABLED, BOTH, YES
from .gui_mp_functions import auto_process_thread
import denmap_src.width_gui as wg
import multiprocessing as mp
import denmap_src.image_encode as image_encode
import base64, io, sys

class denmap_gui:
    from .gui_about import gui_about
    from .gui_file import gui_load_image, gui_update_title, gui_load_points, gui_quit
    from .gui_image import gui_draw_circles, gui_update_image, gui_scale
    from .gui_misc import gui_enable_menu, gui_disable_menu, unset_view_variables, do_nothing
    from .gui_process import gui_auto_process, gui_update_progress, gui_finish_process, gui_fft_thres, create_progressbar, gui_update_fft_progress, gui_binary, gui_NCC
    from .gui_view import gui_change_original, gui_change_binary, gui_change_filt, gui_change_filt_cent
    from .gui_analysis import gui_measure_scale, check_analysis_data, plot_PDAS_hist_axes, gui_PDAS_histogram, plot_N_hist_axes, gui_N_histogram, plot_K_N_axes, gui_K_N, plot_PDAS_N_axes, gui_PDAS_N
    from .gui_analysis_2d_maps import gui_calculate_display_voronoi, gui_calculate_heat_map, image_heat_map
    from .gui_analysis_surface_maps import gui_inverse_surface_map, gui_surface_map, gui_inverse_2d_surface_map, gui_2d_surface_map
    from .gui_zoom import zoom_or_update, gui_zoom, gui_mouse_wheel, gui_control_press, gui_control_release, gui_control_corner
    from .gui_popup_points import create_popup_menu, gui_right_click, popup_add_point, popup_remove_point, unbind_left_click, cont_add_point, cont_add_control_release, bind_cont_add, unbind_cont_add, popup_cont_add_points, popup_highlight
    from .gui_export import write_data_excel, gui_export_images, gui_export_graphs, gui_export_maps, gui_export_everything

    def __init__(self, version_text, year_text):
        self.version_text = version_text
        self.year_text = year_text
        self.impact_logo = Image.open(io.BytesIO(base64.decodebytes(image_encode.img())))
        self.denmap_icon = Image.open(io.BytesIO(base64.decodebytes(image_encode.icon())))
        self.uol_logo = Image.open(io.BytesIO(base64.decodebytes(image_encode.uol_logo())))
        self.uol_logo = self.uol_logo.resize((243,65))
        self.window = Tk()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.window_x_size = self.screen_width//4
        self.window_y_size = 0
        self.window_x_pos = self.screen_width//2-self.window_x_size//2
        self.window_y_pos = 0
        self.FFT_in = False
        self.FFT_out = 100
        self.main_image = False
        self.p_centres = False
        self.circle_image = False
        self.new_image = False
        self.binary_image = False
        self.NCC_values = False
        self.NCC_temp = False
        self.NCC_thres = False
        self.original_image = False
        self.highlight_pts = []
        self.extra_pts = []
        self.analysis_data = False
        self.pixel_scaling = [1,1,1]
        self.load_state = 0
        self.distance_unit = "pixels"
        self.shared_memory = wg.gpu_shared_memory()
        self.zoom_scale = 1
        self.zoom_corner = [0,0]
        self.res_queue = mp.Queue()
        self.prog_queue = mp.Queue()
        self.thr = mp.Process( target=auto_process_thread, args = ( ), daemon=True )
        self.window.title("DenMap \u00A9 "+version_text)

        self.menu = Menu(self.window)

        self.menu_file = Menu(self.menu,tearoff=0)

        self.menu_file.add_command(label='Load',command=self.gui_load_image)
        self.menu_file.add_command(label='Import Points', command=self.gui_load_points)
        self.menu_file.add_command(label='Quit',command=self.gui_quit)

        self.menu_file.entryconfig(1,state=DISABLED)
        self.menu.add_cascade(label='File', menu=self.menu_file)

        self.menu_process = Menu(self.menu,tearoff=0)

        self.menu_process.add_command(label='Auto Process',command=self.gui_auto_process)
        self.menu_process.add_command(label='FFT Bandpass',command=self.gui_fft_thres)
        self.menu_process.add_command(label='Binary Threshold', command=self.gui_binary)
        self.menu_process.add_command(label='NCC Threshold', command=self.gui_NCC)

        self.menu.add_cascade(label='Process', menu=self.menu_process)

        self.menu_view = Menu(self.menu,tearoff=0)
        self.image_checkvariables = [
            BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar()
        ]
        self.menu_view.add_checkbutton(label='Original Image',command=self.gui_change_original, offvalue=0, onvalue=1, variable=self.image_checkvariables[0])
        self.menu_view.add_checkbutton(label='Binary Image',command=self.gui_change_binary, offvalue=0, onvalue=1, variable=self.image_checkvariables[1])
        self.menu_view.add_checkbutton(label='Filtered Image', command=self.gui_change_filt, offvalue=0, onvalue=1, variable=self.image_checkvariables[2])
        self.menu_view.add_checkbutton(label='Filtered Image with Centres', command=self.gui_change_filt_cent, offvalue=0, onvalue=1, variable=self.image_checkvariables[3])

        self.menu.add_cascade(label='View', menu=self.menu_view)


        self.menu_image = Menu(self.menu,tearoff=0)

        self.menu_image.add_command(label='Change Resolution', command=self.gui_scale)
        #menu_image.add_command(label='Rotate')

        self.menu.add_cascade(label='Image', menu=self.menu_image)

        self.menu_analyse = Menu(self.menu,tearoff=0)
        self.menu_analyse.add_command(label='Set Image Scale',command=self.gui_measure_scale)
        self.menu_analyse.add_command(label='Local Spacing Histogram',command=self.gui_PDAS_histogram)
        self.menu_analyse.add_command(label='N Histogram',command=self.gui_N_histogram)
        self.menu_analyse.add_command(label='K vs N',command=self.gui_K_N)
        self.menu_analyse.add_command(label='Local Spacing vs N',command=self.gui_PDAS_N)
        self.menu_analyse.add_command(label='Display Voronoi',command=self.gui_calculate_display_voronoi)
        self.menu_analyse.add_command(label='Display Local Spacing Map',command=self.gui_calculate_heat_map)
        self.menu_analyse.add_command(label='Display 3D Solutally Stable Curvature',command=self.gui_inverse_surface_map)
        self.menu_analyse.add_command(label='Display 3D Solutally Unstable Curvature',command=self.gui_surface_map)
        self.menu_analyse.add_command(label='Display 2D Solutally Stable Curvature',command=self.gui_inverse_2d_surface_map)
        self.menu_analyse.add_command(label='Display 2D Solutally Unstable Curvature',command=self.gui_2d_surface_map)

        self.menu_analyse.entryconfig(0, state="disabled")
        
        self.menu.add_cascade(label='Analyse', menu=self.menu_analyse)

        self.menu_export = Menu(self.menu,tearoff=0)

        self.menu_export.add_command(label='Export All Images',command=self.gui_export_images)
        self.menu_export.add_command(label='Export All Graphs',command=self.gui_export_graphs)
        self.menu_export.add_command(label='Export All Maps',command=self.gui_export_maps)
        self.menu_export.add_command(label='Export Everything',command=self.gui_export_everything)
        self.menu.add_cascade(label='Export', menu=self.menu_export)

        self.menu_help = Menu(self.menu,tearoff=0)

        #menu_help.add_command(label='Github')
        self.menu_help.add_command(label='About DenMap',command=self.gui_about)

        self.menu.add_cascade(label='Help', menu=self.menu_help)

        self.window.config(menu=self.menu)

        self.window.geometry(f"{self.window_x_size}x{self.window_y_size}+{self.window_x_pos}+{self.window_y_pos}")
        self.window.resizable(0,0)

        self.gui_disable_menu()

        self.window_canvas = Canvas(self.window,bd=-2)
        self.window_canvas.pack(fill=BOTH, expand=YES)
        self.create_popup_menu()
        if sys.platform.startswith('win'):
            # On Windows calling this function is necessary.
            mp.freeze_support()
            self.window.bind("<Control-MouseWheel>", self.gui_mouse_wheel)
            self.window.bind("<Button-3>",self.gui_right_click)
            self.window.bind("<Control-ButtonPress-1>",self.gui_control_press)
            self.window.bind("<Control-ButtonRelease-1>",self.gui_control_release)
        else:
            self.window.bind("<Control-Button-4>", self.gui_mouse_wheel)
            self.window.bind("<Control-Button-5>", self.gui_mouse_wheel)
            self.window.bind("<Button-3>",self.gui_right_click)
            self.window.bind("<Control-ButtonPress-1>",self.gui_control_press)
            self.window.bind("<Control-ButtonRelease-1>",self.gui_control_release)

        self.window.iconphoto(True, ImageTk.PhotoImage(self.denmap_icon))
        self.window.mainloop()