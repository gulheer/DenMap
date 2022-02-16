from tkinter import Toplevel, Button, Checkbutton, Canvas, Label, Frame, Menu, Entry, Scale, IntVar, StringVar, BooleanVar, messagebox, filedialog, TOP, NW, END, LEFT, RIGHT, BOTH, N, S, W, E, HORIZONTAL, YES, CENTER, DISABLED
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2, xlsxwriter, os
import numpy as np
import matplotlib.pyplot as plt


def gui_export_images(self):
    export_images_window = Toplevel()
    export_images_window.withdraw()
    export_images_window.title("Export Images")
    export_images_window.transient(self.window)
    export_images_window.grab_set()
    def export_images_close():
        export_images_window.destroy()
    export_images_window.protocol("WM_DELETE_WINDOW", export_images_close)
    f1 = Frame(export_images_window)
    f1.grid(row=0, column=0, columnspan=3, sticky=N+S+E+W)
    fft = IntVar()
    fft_check = Checkbutton(f1, text="Filtered Image", variable=fft)
    fft_check.pack(side=TOP, anchor='w')
    orig_mapped = IntVar()
    orig_mapped_check = Checkbutton(f1, text="Original Image Mapped", variable=orig_mapped)
    orig_mapped_check.pack(side=TOP, anchor='w')
    fft_mapped = IntVar()
    fft_mapped_check = Checkbutton(f1, text="Filtered Image Mapped", variable=fft_mapped)
    fft_mapped_check.pack(side=TOP, anchor='w')
    data_points = IntVar()
    data_points_check = Checkbutton(f1, text="Data points file", variable=data_points)
    data_points_check.pack(side=TOP, anchor='w')
    if self.load_state == 0:
        if type(self.new_image) == bool:
            fft_check.config(state=DISABLED)
            fft_mapped_check.config(state=DISABLED)
        if type(self.p_centres) == bool:
            orig_mapped_check.config(state=DISABLED)
            data_points_check.config(state=DISABLED)
    else:
        fft_check.config(state=DISABLED)
        fft_mapped_check.config(state=DISABLED)
    def export_gui():
        ticked = (fft.get() or orig_mapped.get() or fft_mapped.get() or data_points.get())
        if ticked:
            img_save_name = filedialog.asksaveasfilename(title="Select the prefix for the files",initialfile = os.path.splitext(os.path.basename(self.img_name))[0],filetypes=[("Tagged Image File Format",".tiff"),("Portable Network Graphics",".png")],defaultextension=".tiff")
            if (img_save_name):
                save_base_name = os.path.splitext(img_save_name)[0]
                save_type = os.path.splitext(img_save_name)[1]
                try:
                    if fft.get():
                        if len(self.new_image.shape) < 3:
                            cv2.imwrite(save_base_name+"_filtered"+save_type,self.new_image)
                        else:
                            cv2.imwrite(save_base_name+"_filtered"+save_type,self.new_image[:,:,0])
                    if orig_mapped.get():
                        if self.load_state == 0:
                            mapped_image = np.array(gui_draw_circles(Image.fromarray(self.original_image),self.p_centres))
                            cv2.imwrite(save_base_name+"_mapped"+save_type,mapped_image[:,:,::-1])
                        else:
                            cv2.imwrite(save_base_name+"_mapped"+save_type,np.array(self.circle_image)[:,:,::-1])
                    if fft_mapped.get():
                        cv2.imwrite(save_base_name+"_filtered_mapped"+save_type,np.array(self.circle_image)[:,:,::-1])
                    if data_points.get():
                        with open(save_base_name+"_data.csv","w") as f:
                            if self.distance_unit != "pixels":
                                f.write(("x (pixels),y (pixels),x ("+self.distance_unit.replace(u'\u03BC','u')+"),y ("+self.distance_unit.replace(u'\u03BC','u')+"),extra"))
                                for p in self.p_centres:
                                    p_scale = p*self.pixel_scaling[2]
                                    f.write(("\n"+str(p[0])+","+str(p[1])+","+str(p_scale[0])+","+str(p_scale[1])+",0"))
                                for p in np.array(self.extra_pts):
                                    p_scale = p*self.pixel_scaling[2]
                                    f.write(("\n"+str(p[0])+","+str(p[1])+","+str(p_scale[0])+","+str(p_scale[1])+",1"))
                            else:
                                f.write("x (pixels),y (pixels),extra")
                                for p in self.p_centres:
                                    f.write(("\n"+str(p[0])+","+str(p[1])+",0"))
                                for p in np.array(self.extra_pts):
                                    f.write(("\n"+str(p[0])+","+str(p[1])+",1"))
                except PermissionError:
                    if path.isfile(img_save_name):
                        messagebox.showerror("Permission Denied","Unable to save image. Please check if image is open.")
                    else:
                        messagebox.showerror("Permission Denied","Unable to save image. Please check if folder exists and has write permissions.")
                except:
                    messagebox.showerror("Unable to export files","Unable to export files!")
                else:
                    messagebox.showinfo("Success!","File(s) have been exported sucessfully!")
                    export_images_close()
        else:
            messagebox.showerror("No files have been selected!","Please select files to export.")
    def select_gui():
        if self.load_state == 0:
            if not type(self.new_image) == bool:
                fft.set(1)
            if not type(self.p_centres) == bool:
                orig_mapped.set(1)
                fft_mapped.set(1)
                data_points.set(1)
        else:
            orig_mapped.set(1)
            data_points.set(1)
    Button(export_images_window,text="Export",command=export_gui).grid(row=1,column=2,padx=2,pady=2)
    Button(export_images_window,text="Select All",command=select_gui).grid(row=1,column=1,padx=2,pady=2)
    Button(export_images_window,text="Cancel",command=export_images_close).grid(row=1,column=0,padx=2,pady=2)
    export_images_window.update()
    export_images_window.geometry(f"+{(self.screen_width-export_images_window.winfo_width())//2}+{(self.screen_height-export_images_window.winfo_height())//2}")
    export_images_window.deiconify()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def write_data_excel(self, filename,Gaussian_coefs,Linear_fits,img_name=False):
	workbook = xlsxwriter.Workbook(filename)
	worksheet = workbook.add_worksheet()

	merge_format = workbook.add_format({'bold': True, 'border':2,'align':'center','valign':'vcenter', 'text_wrap':True})
	max_n = max(self.analysis_data.neighs[:,1])
	worksheet.merge_range(0 ,0 , 0, 1, 'Units', merge_format)
	worksheet.merge_range(0,2,0,max_n+1, '('+self.distance_unit+')', merge_format)
	worksheet.merge_range(1,0,1,max_n+1, '', merge_format)
	cell_titles = ["Parameter","Symbol","Bulk","Boundary"]

	for i in range(0,len(cell_titles)):
		worksheet.write(2,i,cell_titles[i],merge_format)

	column_widths = [8.43]*(max_n+2)
	column_widths[0] = 14
	for i in range(len(column_widths)):
		worksheet.set_column(i,i,column_widths[i])

	subscript = workbook.add_format({'bold': True, 'font_script': 2})

	for i in range(0,max_n-2):
		worksheet.write_rich_string(2,len(cell_titles)+i,merge_format,"N",subscript,str(i+3),merge_format)

	italic = workbook.add_format({'italic': True, 'border':2,'align':'center', 'valign':'vcenter','text_wrap': True})

	worksheet.write(3,0,"Number of cells/dendrites detected",italic)
	worksheet.write(4,0,"The mean array spacing",italic)
	worksheet.write(5,0,"The local primary spacing standard deviation",italic)
	worksheet.write(6,0,"The mean local primary spacing for each coordination number",italic)
	worksheet.write(7,0,"The local primary spacing standard deviation for each coordination number",italic)
	worksheet.write(8,0,"The mean array packing factor",italic)
	worksheet.write(9,0,"The standard deviation of array packing factor",italic)

	worksheet.write(10,0,"The mean packing factor for each coordination number",italic)
	worksheet.write(11,0,"The standard deviation of packing factor for each coordination number",italic)
	worksheet.write(12,0,"The mean local primary spacing range for the array",italic)
	worksheet.write(13,0,"The standard deviation of mean local primary spacing range",italic)
	worksheet.write(14,0,"The mean local primary spacing range for each coordination number",italic)
	worksheet.write(15,0,"The standard deviation of mean local primary spacing range for each coordination number",italic)
	worksheet.write(16,0,"The mean array coordination number",italic)
	worksheet.write(17,0,"The standard deviation of array coordination number",italic)

	default = workbook.add_format({'border':2,'align':'center', 'valign':'vcenter','text_wrap': True})

	symbols_list = [["-"],
					[u"\u03BB\u0305",subscript,"Array"],
					[u"\u03C3",subscript,u"\u03BB Array"],
					[u"\u03BB\u0305",subscript,"N"],
					[u"\u03C3",subscript,u"\u03BB N"],
					[u"K",subscript,"Array"],
					[u"\u03C3",subscript,"K"],
					[u"K\u0305",subscript,"N"],
					[u"\u03C3",subscript,"K N"],
					["MLPR"],
					[u"\u03C3",subscript,"MLPR"],
					["MLPR",subscript,"N"],
					[u"\u03C3",subscript,"MLPR N"],
					[u"N\u0305",subscript,"Array"],
					[u"\u03C3",subscript,"N"]
					]

	for i in range(len(symbols_list)):
		if len(symbols_list[i]) > 1:
			worksheet.write_rich_string(i+3,1,*(symbols_list[i]+[italic]))
		else:
			worksheet.write(i+3,1,*(symbols_list[i]+[italic]))

	worksheet.write(3,2,len(self.analysis_data.points),default)
	worksheet.write(3,3,len(self.analysis_data.boundary_centres),default)

	neigh_counts = []

	for i in range(0,max_n-2):
		neigh_counts += [np.sum(self.analysis_data.neighs[:,1] == i+3)]
		worksheet.write(3,len(cell_titles)+i,neigh_counts[-1],default)

	worksheet.write(4,2,self.analysis_data.pdas_mean,default)
	b_spacings = np.concatenate(self.analysis_data.boundary_spacings)
	worksheet.write(4,3,np.mean(b_spacings),default)

	for i in range(0,max_n-2):
		worksheet.write(4,len(cell_titles)+i,"-",default)

	worksheet.write(5,2,self.analysis_data.pdas_std,default)
	worksheet.write(5,3,np.std(b_spacings),default)

	for i in range(0,max_n-2):
		worksheet.write(5,len(cell_titles)+i,"-",default)

	neigh_pos = np.insert(np.cumsum(neigh_counts),0,0)

	worksheet.write(6,2,"-",default)
	worksheet.write(6,3,"-",default)

	for i in range(0,max_n-2):
		if len(self.analysis_data.spacings[neigh_pos[i]:neigh_pos[i+1]]) > 0:
			worksheet.write(6,len(cell_titles)+i,np.mean(np.concatenate(self.analysis_data.spacings[neigh_pos[i]:neigh_pos[i+1]])),default)
		else:
			worksheet.write(6,len(cell_titles)+i,"-",default)

	worksheet.write(7,2,"-",default)
	worksheet.write(7,3,"-",default)

	for i in range(0,max_n-2):
		if len(self.analysis_data.spacings[neigh_pos[i]:neigh_pos[i+1]]) > 0:
			worksheet.write(7,len(cell_titles)+i,np.std(np.concatenate(self.analysis_data.spacings[neigh_pos[i]:neigh_pos[i+1]])),default)
		else:
			worksheet.write(7,len(cell_titles)+i,"-",default)

	worksheet.write(8,2,np.mean(np.concatenate(self.analysis_data.K)),default)
	worksheet.write(8,3,"-",default)

	for i in range(0,max_n-2):
		worksheet.write(8,len(cell_titles)+i,"-",default)

	worksheet.write(9,2,np.std(np.concatenate(self.analysis_data.K)),default)
	worksheet.write(9,3,"-",default)

	for i in range(0,max_n-2):
		worksheet.write(9,len(cell_titles)+i,"-",default)

	worksheet.write(10,2,"-",default)
	worksheet.write(10,3,"-",default)

	for i in range(0,max_n-2):
		res = self.analysis_data.K_mean[i]
		if np.isnan(res) or np.isinf(res):
			worksheet.write(10,len(cell_titles)+i,"-",default)
		else:
			worksheet.write(10,len(cell_titles)+i,res,default)

	worksheet.write(11,2,"-",default)
	worksheet.write(11,3,"-",default)

	for i in range(0,max_n-2):
		res = self.analysis_data.K_std[i]
		if np.isnan(res) or np.isinf(res):
			worksheet.write(11,len(cell_titles)+i,"-",default)
		else:
			worksheet.write(11,len(cell_titles)+i,res,default)
	
	worksheet.write(12,2,np.mean(self.analysis_data.pdas_range),default)
	worksheet.write(12,3,np.mean(self.analysis_data.boundary_range[self.analysis_data.boundary_range>0]),default)

	for i in range(0,max_n-2):
		worksheet.write(12,len(cell_titles)+i,"-",default)

	worksheet.write(13,2,np.std(self.analysis_data.pdas_range),default)
	worksheet.write(13,3,np.std(self.analysis_data.boundary_range[self.analysis_data.boundary_range>0]),default)

	for i in range(0,max_n-2):
		worksheet.write(13,len(cell_titles)+i,"-",default)

	worksheet.write(14,2,"-",default)
	worksheet.write(14,3,"-",default)

	for i in range(0,max_n-2):
		if len(self.analysis_data.spacings[neigh_pos[i]:neigh_pos[i+1]]) > 0:
			worksheet.write(14,len(cell_titles)+i,np.mean(self.analysis_data.pdas_range[neigh_pos[i]:neigh_pos[i+1]]),default)
		else:
			worksheet.write(14,len(cell_titles)+i,"-",default)

	worksheet.write(15,2,"-",default)
	worksheet.write(15,3,"-",default)

	for i in range(0,max_n-2):
		if len(self.analysis_data.spacings[neigh_pos[i]:neigh_pos[i+1]]) > 0:
			worksheet.write(15,len(cell_titles)+i,np.std(self.analysis_data.pdas_range[neigh_pos[i]:neigh_pos[i+1]]),default)
		else:
			worksheet.write(15,len(cell_titles)+i,"-",default)

	w_N_mean,w_N_std = weighted_avg_and_std(np.arange(3,max_n+1), neigh_counts)

	worksheet.write(16,2,w_N_mean,default)
	worksheet.write(17,2,w_N_std,default)
	worksheet.write(16,3,"-",default)
	worksheet.write(17,3,"-",default)

	for i in range(0,max_n-2):
		worksheet.write(16,len(cell_titles)+i,"-",default)
		worksheet.write(17,len(cell_titles)+i,"-",default)

	if self.analysis_data.porosity:
		worksheet.write(18,0,"Number of pores detected",italic)
		worksheet.write(19,0,"Average array number of pores per Voronoi area",italic)
		worksheet.write(20,0,"The standard deviation of average array number of pores per Voronoi area",italic)
		worksheet.write(21,0,"Average number of pores per Voronoi area for each coordination number",italic)
		worksheet.write(22,0,"The standard deviation of number of pores per Voronoi area for each coordination number",italic)
		worksheet.write(23,0,"Average array pore area",italic)
		worksheet.write(24,0,"The standard deviation of average array pore area",italic)
		worksheet.write(25,0,"Average array total pore area",italic)
		worksheet.write(26,0,"The standard deviation of the array total pore area",italic)
		worksheet.write(27,0,"Average total pore area for each coordination number",italic)
		worksheet.write(28,0,"The standard deviation of the total pore area for each coordination number",italic)
		worksheet.write(29,0,"Normalised Average array normalised total pore area",italic)
		worksheet.write(30,0,"The standard deviation of array normalised total pore area",italic)
		worksheet.write(31,0,"Normalised Average total pore area for each coordination number",italic)
		worksheet.write(32,0,"The standard deviation of normalised total pore area for each coordination number",italic)
		porosity_list = [
						['-'],
						[default,u'n\u0305',subscript,"Array"],
						[default,u"\u03C3",subscript,"n Array"],
						[default,u'n\u0305',subscript,'N'],
						[default,u"\u03C3",subscript,"n N"],
						[default,u"\u03C6\u0305",subscript,"Array"],
						[u"\u03C3",subscript,u"\u03C6"],
						[default,u"\u03C6\u0305",subscript,"Array Total"],
						[u"\u03C3",subscript,u"\u03C6 Total"],
						[default,u"\u03C6\u0305",subscript,"N"],
						[u"\u03C3",subscript,u"\u03C6 N"],
						[default,u"\u03C6\u0305",subscript,"Norm Array"],
						[u"\u03C3",subscript,"\u03C6 Norm Array"],
						[default,u"\u03C6\u0305",subscript,"Norm N"],
						[u"\u03C3",subscript,"\u03C6 Norm N"]
		]
		offset_porosity = len(symbols_list)
		for i in range(len(porosity_list)):
			if len(porosity_list[i]) > 1:
				worksheet.write_rich_string(i+3+offset_porosity,1,*(porosity_list[i]+[italic]))
			else:
				worksheet.write(i+3+offset_porosity,1,*(porosity_list[i]+[italic]))

		worksheet.write(18,2,len(self.analysis_data.sample_hole_areas),default)
		worksheet.write(19,2,self.analysis_data.porosity_mean_n,default)
		worksheet.write(20,2,self.analysis_data.porosity_std_n,default)
		worksheet.write(21,2,'-',default)
		worksheet.write(22,2,'-',default)

		worksheet.write(18,3,self.analysis_data.sample_n_holes[0] if self.analysis_data.sample_n_holes[0] > 0 else "-",default)
		worksheet.write(19,3,self.analysis_data.porosity_mean_n_N[0] if self.analysis_data.porosity_mean_n_N[0] > 0 else "-",default)
		worksheet.write(20,3,self.analysis_data.porosity_std_n_N[0] if self.analysis_data.porosity_std_n_N[0] > 0 else "-",default)
		worksheet.write(21,3,'-',default)
		worksheet.write(22,3,'-',default)

		for i in range(0,max_n-2):
			worksheet.write(18,len(cell_titles)+i,self.analysis_data.sample_n_holes[i+1],default)
			worksheet.write(19,len(cell_titles)+i,'-',default)
			worksheet.write(20,len(cell_titles)+i,'-',default)
			worksheet.write(21,len(cell_titles)+i,self.analysis_data.porosity_mean_n_N[i+1] if self.analysis_data.porosity_mean_n_N[i+1] > 0 else "-",default)
			worksheet.write(22,len(cell_titles)+i,self.analysis_data.porosity_std_n_N[i+1] if self.analysis_data.porosity_mean_n_N[i+1] > 0 else "-",default)
		
		worksheet.write(23,2,np.mean(self.analysis_data.sample_hole_areas),default)
		worksheet.write(24,2,np.std(self.analysis_data.sample_hole_areas),default)
		worksheet.write(25,2,self.analysis_data.porosity_mean_total_area,default)
		worksheet.write(26,2,self.analysis_data.porosity_std_total_area,default)
		worksheet.write(27,2,"-",default)
		worksheet.write(28,2,"-",default)

		worksheet.write(23,3,"-",default)
		worksheet.write(24,3,"-",default)
		worksheet.write(25,3,self.analysis_data.porosity_mean_total_area_per_N[0] if self.analysis_data.porosity_mean_total_area_per_N[i+1] > 0 else "-",default)
		worksheet.write(26,3,self.analysis_data.porosity_std_total_area_per_N[0] if self.analysis_data.porosity_std_total_area_per_N[i+1] > 0 else "-",default)
		worksheet.write(27,3,"-",default)
		worksheet.write(28,3,"-",default)

		for i in range(0,max_n-2):
			worksheet.write(23,len(cell_titles)+i,"-",default)
			worksheet.write(24,len(cell_titles)+i,"-",default)
			worksheet.write(25,len(cell_titles)+i,"-",default)
			worksheet.write(26,len(cell_titles)+i,"-",default)
			worksheet.write(27,len(cell_titles)+i,self.analysis_data.porosity_mean_total_area_per_N[i+1] if self.analysis_data.porosity_mean_total_area_per_N[i+1] > 0 else "-",default)
			worksheet.write(28,len(cell_titles)+i,self.analysis_data.porosity_std_total_area_per_N[i+1] if self.analysis_data.porosity_mean_total_area_per_N[i+1] > 0 else "-",default)
		
		worksheet.write(29,2,self.analysis_data.porosity_mean_normalised_area ,default)
		worksheet.write(30,2,self.analysis_data.porosity_std_normalised_area ,default)
		worksheet.write(31,2,"-",default)
		worksheet.write(32,2,"-",default)
		
		worksheet.write(29,3,self.analysis_data.porosity_mean_normalised_area_per_N[0],default)
		worksheet.write(30,3,self.analysis_data.porosity_mean_normalised_area_per_N[0],default)
		worksheet.write(31,3,"-",default)
		worksheet.write(32,3,"-",default)

		for i in range(0,max_n-2):
			worksheet.write(29,len(cell_titles)+i,"-",default)
			worksheet.write(30,len(cell_titles)+i,"-",default)
			worksheet.write(31,len(cell_titles)+i,self.analysis_data.porosity_mean_normalised_area_per_N[i+1] if self.analysis_data.porosity_mean_normalised_area_per_N[i+1] > 0 else "-",default)
			worksheet.write(32,len(cell_titles)+i,self.analysis_data.porosity_std_normalised_area_per_N[i+1] if self.analysis_data.porosity_mean_normalised_area_per_N[i+1] > 0 else "-",default)


	col = len(cell_titles)+max_n-1
	worksheet.merge_range(0 ,col , 1, col, u'\u03BB histogram Gaussian fit:', merge_format)
	worksheet.set_column(col,col,len('Gaussian Fit:')+1)
	worksheet.write(0,col+1,"a", merge_format)
	worksheet.write(0,col+2,u"\u03BC", merge_format)
	worksheet.write(0,col+3,u"\u03C3", merge_format)
	worksheet.write(1,col+1,Gaussian_coefs[0], default)
	worksheet.write(1,col+2,Gaussian_coefs[1], default)
	worksheet.write(1,col+3,Gaussian_coefs[2], default)

	worksheet.write(3,col,"K vs N linear fit", merge_format)
	worksheet.write(3,col+1,"m", merge_format)
	worksheet.write(3,col+2,"c", merge_format)
	worksheet.write(3,col+3,"R-value", merge_format)
	worksheet.write(3,col+4,"Std Error", merge_format)

	worksheet.write(4,col,"Max", merge_format)
	worksheet.write(4,col+1,Linear_fits[0].slope, default)
	worksheet.write(4,col+2,Linear_fits[0].intercept, default)
	worksheet.write(4,col+3,Linear_fits[0].rvalue, default)
	worksheet.write(4,col+4,Linear_fits[0].stderr, default)

	worksheet.write(5,col,"Mean", merge_format)
	worksheet.write(5,col+1,Linear_fits[1].slope, default)
	worksheet.write(5,col+2,Linear_fits[1].intercept, default)
	worksheet.write(5,col+3,Linear_fits[1].rvalue, default)
	worksheet.write(5,col+4,Linear_fits[1].stderr, default)

	worksheet.write(6,col,"Min", merge_format)
	worksheet.write(6,col+1,Linear_fits[1].slope, default)
	worksheet.write(6,col+2,Linear_fits[1].intercept, default)
	worksheet.write(6,col+3,Linear_fits[1].rvalue, default)
	worksheet.write(6,col+4,Linear_fits[1].stderr, default)

	if type(img_name) != bool:
		worksheet.insert_image(7,col, img_name)
	workbook.close()

def gui_export_graphs(self):
	if not self.check_analysis_data():
		return
	export_graphs_window = Toplevel()
	export_graphs_window.withdraw()
	def export_graphs_close():
		export_graphs_window.destroy()
		plt.close('all')
	export_graphs_window.protocol("WM_DELETE_WINDOW", export_graphs_close)
	export_graphs_window.title("Export Graphs")
	export_graphs_window.transient(self.window)
	export_graphs_window.grab_set()
	f = plt.figure(figsize=(10,8))
	canvas = FigureCanvasTkAgg(f, master=export_graphs_window)
	#canvas.get_tk_widget().pack(fill=BOTH, expand=1)
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=2)
	ax = f.add_subplot(2,2,1)
	ax2 = f.add_subplot(2,2,2)
	ax3 = f.add_subplot(2,2,3)
	ax4 = f.add_subplot(2,2,4)
	coef = self.plot_PDAS_hist_axes(ax)
	self.plot_K_N_axes(ax2)
	self.plot_N_hist_axes(ax3)
	fit_mean, fit_min, fit_max = self.plot_PDAS_N_axes(ax4)
	ax.margins(0,tight=True)
	f.subplots_adjust(left=0.06,right=0.99,bottom=0.06,top=0.98)
	f.canvas.draw()
	export_graphs_window.update()
	def only_numbers(char):
		return char.isdigit()
	vcmd = (export_graphs_window.register(only_numbers),'%S')
	f1 = Frame(export_graphs_window)
	f1.grid(row=1, column=0, sticky=N+S+E+W)
	dpi_label = Label(f1, text="Dots per inch (DPI):",justify=LEFT)
	dpi_label.pack(side=LEFT)
	dpi = StringVar()
	dpi.set(str(f.dpi))
	Entry(f1,width=5,validate="key",validatecommand=vcmd, textvariable = dpi).pack(side=LEFT,padx=5)
	res_text = StringVar()
	res = (f.get_size_inches()*float(dpi.get())).astype(int)
	res_text.set(f"Expected Resolution: {res[0]}x{res[1]}")
	Label(f1, textvariable=res_text).pack(side=LEFT,padx=5)
	def resolution_label_change(text):
		res = (f.get_size_inches()*float(dpi.get())).astype(int)
		res_text.set(f"Expected Resolution: {res[0]}x{res[1]}")
	dpi.trace("w", lambda *args: resolution_label_change(res_text))
	def export_graphs():
		img_save_name = filedialog.asksaveasfilename(title="Select file to export all graphs to",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			data_save_name = os.path.splitext(img_save_name)[0]+".xlsx"
			f.savefig(img_save_name,dpi=float(dpi.get()))
			if data_export.get():
				self.write_data_excel(data_save_name,coef,[fit_max,fit_mean,fit_min])
	f2 = Frame(export_graphs_window)
	f2.grid(row=1, column=1, sticky=N+S+E+W)
	data_export = IntVar()
	data_export.set(1)
	data_check = Checkbutton(f2, text="Export data", variable=data_export)
	Button(f2,text="Export",command=export_graphs).pack(side=RIGHT,pady=5,padx=5)
	data_check.pack(side=LEFT)
	export_graphs_window.update()
	export_graphs_window.geometry(f"+{(self.screen_width-export_graphs_window.winfo_width())//2}+{(self.screen_height-export_graphs_window.winfo_height())//2}")
	export_graphs_window.deiconify()

def gui_export_maps(self):
    if not self.check_analysis_data():
        return
    export_maps_window = Toplevel()
    export_maps_window.withdraw()
    export_maps_window.title("Export Maps")
    export_maps_window.transient(self.window)
    export_maps_window.grab_set()
    def export_maps_close():
        export_maps_window.destroy()
    export_maps_window.protocol("WM_DELETE_WINDOW", export_maps_close)
    f1 = Frame(export_maps_window)
    f1.grid(row=0, column=0, columnspan=3, sticky=N+S+E+W)
    voronoi = IntVar()
    voronoi_check = Checkbutton(f1, text="Voronoi Map", variable=voronoi)
    voronoi_check.pack(side=TOP, anchor='w')
    pdas_spacing = IntVar()
    pdas_spacing_check = Checkbutton(f1, text="Local Spacing Map", variable=pdas_spacing)
    pdas_spacing_check.pack(side=TOP, anchor='w')
    def export_gui():
        ticked = (voronoi.get() or pdas_spacing.get())
        if ticked:
            img_save_name = filedialog.asksaveasfilename(title="Select the prefix for the files",initialfile = os.path.splitext(os.path.basename(self.img_name))[0],filetypes=[("Portable Network Graphics",".png"),("Tagged Image File Format",".tiff")],defaultextension=".png")
            if (img_save_name):
                img_save_name = os.path.splitext(img_save_name)[0]
                if type(self.new_image) == bool:
                    image = self.original_image
                else:
                    image = self.new_image
                try:
                    if voronoi.get():
                    	voronoi_image = self.analysis_data.generate_voronoi_image(image,show_points=False,color_bar=True)
                    	cv2.imwrite(img_save_name+"_voronoi.png",voronoi_image[:,:,::-1])
                    if pdas_spacing.get():
                    	if type(self.analysis_data.nn_interpolated_values) == bool:
                    		nn_interpolated_values,truth_grid = self.analysis_data.calc_heat_map(self.main_image)
                    	else:
                    		nn_interpolated_values = self.analysis_data.nn_interpolated_values
                    		truth_grid = self.analysis_data.nn_truth_grid
                    	heat_image = self.image_heat_map(image,nn_interpolated_values,truth_grid,show_points=True)
                    	cv2.imwrite(img_save_name+"_local_spacing.png",heat_image[:,:,::-1])
                except PermissionError:
                    if path.isfile(img_save_name):
                        messagebox.showerror("Permission Denied","Unable to save image. Please check if image is open.")
                    else:
                        messagebox.showerror("Permission Denied","Unable to save image. Please check if folder exists and has write permissions.")
                except:
                    messagebox.showerror("Failed!","Unable to save file(s)!")
                else:
                    messagebox.showinfo("Success!","File(s) have been exported sucessfully!")
                    export_maps_close()
        else:
            messagebox.showerror("No files have been selected!","Please select files to export.")
    def select_gui():
        voronoi.set(1)
        pdas_spacing.set(1)
    Button(export_maps_window,text="Export",command=export_gui).grid(row=1,column=2,padx=2,pady=2)
    Button(export_maps_window,text="Select All",command=select_gui).grid(row=1,column=1,padx=2,pady=2)
    Button(export_maps_window,text="Cancel",command=export_maps_close).grid(row=1,column=0,padx=2,pady=2)
    export_maps_window.update()
    export_maps_window.geometry("+"+str((self.screen_width-export_maps_window.winfo_width())//2)+"+"+str((self.screen_height-export_maps_window.winfo_height())//2))
    export_maps_window.deiconify()

def gui_export_everything(self):
    img_save_name = filedialog.asksaveasfilename(title="Select the prefix for the files",initialfile = os.path.splitext(os.path.basename(self.img_name))[0],filetypes=[('Portable Network Graphics','.png'),("Tagged Image File Format",".tiff")],defaultextension=".png")
    if (img_save_name):
        try:
            save_base_name = os.path.splitext(img_save_name)[0]
            save_type = os.path.splitext(img_save_name)[1]
            global analysis_data, circle_image, highlight_pts
            if not self.check_analysis_data():
                return
            f = plt.figure(figsize=(10,8))
            ax = f.add_subplot(2,2,1)
            ax2 = f.add_subplot(2,2,2)
            ax3 = f.add_subplot(2,2,3)
            ax4 = f.add_subplot(2,2,4)
            coef = self.plot_PDAS_hist_axes(ax)
            self.plot_K_N_axes(ax2)
            self.plot_N_hist_axes(ax3)
            fit_mean, fit_min, fit_max = self.plot_PDAS_N_axes(ax4)
            ax.margins(0,tight=True)
            f.subplots_adjust(left=0.06,right=0.99,bottom=0.06,top=0.98)
            f.dpi = 300
            f.canvas.draw()
            all_graphs = np.array(f.canvas.renderer.buffer_rgba())[:,:,0:3]
            plt.close('all')

            self.write_data_excel(save_base_name+"_results.xlsx",coef,[fit_max,fit_mean,fit_min])
            cv2.imwrite(save_base_name+"_graphs.png",all_graphs[:,:,::-1])
            if type(self.new_image) != bool:
                if len(self.new_image.shape) < 3:
                    cv2.imwrite(save_base_name+"_filtered"+save_type,self.new_image)
                else:
                    cv2.imwrite(save_base_name+"_filtered"+save_type,self.new_image[:,:,0])
                if type(self.circle_image) != bool:
                    if self.highlight_outliers.get():
                        self.highlight_pts = []
                        self.circle_image = self.gui_draw_circles(self.original_image,self.p_centres)
                    cv2.imwrite(save_base_name+"_filtered_mapped"+save_type,np.array(self.circle_image)[:,:,::-1])
            if type(self.original_image) != bool:
                if self.load_state == 0:
                    if self.highlight_outliers.get():
                        self.highlight_pts = []
                    mapped_image = gui_draw_circles(self.original_image,self.p_centres,return_np=True)
                    cv2.imwrite(save_base_name+"_mapped"+save_type,mapped_image[:,:,::-1])
                else:
                    if self.highlight_outliers.get():
                        self.highlight_pts = []
                        self.circle_image = gui_draw_circles(self.original_image,self.p_centres)
                    cv2.imwrite(save_base_name+"_mapped"+save_type,np.array(self.circle_image)[:,:,::-1])
            self.highlight_outliers.set(0)
            with open(save_base_name+"_data.csv","w") as f:
                if self.distance_unit != "pixels":
                    f.write(("x (pixels),y (pixels),x ("+self.distance_unit.replace(u'\u03BC','u')+"),y ("+self.distance_unit.replace(u'\u03BC','u')+"),extra"))
                    for p in self.p_centres:
                        p_scale = p*self.pixel_scaling[2]
                        f.write(("\n"+str(p[0])+","+str(p[1])+","+str(p_scale[0])+","+str(p_scale[1])+",0"))
                    for p in np.array(self.extra_pts):
                        p_scale = p*self.pixel_scaling[2]
                        f.write(("\n"+str(p[0])+","+str(p[1])+","+str(p_scale[0])+","+str(p_scale[1])+",1"))
                else:
                    f.write("x (pixels),y (pixels),extra")
                    for p in self.p_centres:
                        f.write(("\n"+str(p[0])+","+str(p[1])+",0"))
                    for p in np.array(self.extra_pts):
                        f.write(("\n"+str(p[0])+","+str(p[1])+",1"))
            if type(self.analysis_data.nn_interpolated_values) == bool:
                nn_interpolated_values,truth_grid = self.analysis_data.calc_heat_map(self.main_image)
            else:
                nn_interpolated_values = self.analysis_data.nn_interpolated_values
                truth_grid = self.analysis_data.nn_truth_grid
            if type(self.new_image) != bool:
                voronoi_image = self.analysis_data.generate_voronoi_image(self.new_image,show_points=False,color_bar=True)
                heat_image = self.image_heat_map(self.new_image,nn_interpolated_values,truth_grid,show_points=True)
            else:
                voronoi_image = self.analysis_data.generate_voronoi_image(self.original_image,show_points=False,color_bar=True)
                heat_image = self.image_heat_map(self.original_image,nn_interpolated_values,truth_grid,show_points=True)
            cv2.imwrite(save_base_name+"_voronoi"+save_type,voronoi_image[:,:,::-1])
            cv2.imwrite(save_base_name+"_local_spacing"+save_type,heat_image[:,:,::-1])
        except:
        	messagebox.showerror("Failed!","Unable to save file(s)!")
        else:
        	messagebox.showinfo("Success!","File(s) have been exported sucessfully!")
