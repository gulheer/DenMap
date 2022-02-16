from tkinter import Toplevel, Button, Checkbutton, Canvas, Label, Frame, Menu, Entry, IntVar, StringVar, BooleanVar, filedialog, NW, RIGHT, BOTH, N, S, W, E, YES, CENTER
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.linalg import lstsq
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import denmap_src.width_gui as wg
import denmap_src.denmap_stats as denmap_stats
import cv2


def quadratic_solv(x,y,z):
	coefs, _, _, _ = lstsq(np.array([ x**2, y**2, x, y,np.ones_like(x)]).T, z.T)
	return coefs

def gui_inverse_surface_map(self):
	if not self.check_analysis_data():
		return
	
	if type(self.analysis_data.pdas_strain) == bool:
		pdas_strain = self.analysis_data.calc_pdas_strain()
	else:
		pdas_strain = self.analysis_data.pdas_strain
	surface_inverse_map_window = Toplevel()
	surface_inverse_map_window.withdraw()
	ax = False
	surface_inverse_map_window.title("3D Solutally Stable Curvature")
	surface_inverse_map_window.transient(self.window)
	surface_inverse_map_window.grab_set()
	surface_inverse_map_window.resizable(0,0)
	inverse_scale_window = Toplevel(surface_inverse_map_window)
	inverse_scale_window.withdraw()
	inverse_range_window = Toplevel(surface_inverse_map_window)
	inverse_range_window.withdraw()
	def inverse_scale_close():
		inverse_scale_window.transient(self.window)
		inverse_scale_window.withdraw()
	def inverse_range_close():
		inverse_range_window.transient(self.window)
		inverse_range_window.withdraw()
	def surface_inverse_map_close():
		inverse_scale_window.destroy()
		inverse_range_window.destroy()
		surface_inverse_map_window.destroy()
		nonlocal ax
		del ax
		plt.close('all')
	surface_inverse_map_window.protocol("WM_DELETE_WINDOW",surface_inverse_map_close)
	inverse_scale_window.protocol("WM_DELETE_WINDOW", inverse_scale_close)
	inverse_scale_window.title("Set Z scale")
	inverse_scale_window.transient(self.window)
	inverse_scale_window.resizable(0,0)
	inverse_range_window.protocol("WM_DELETE_WINDOW", inverse_range_close)
	inverse_range_window.title("Set Z range")
	inverse_range_window.transient(self.window)
	inverse_range_window.resizable(0,0)
	f = plt.figure(figsize=(5,4))
	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	f.set_size_inches(im_w/f.dpi,im_h/f.dpi)
	canvas = FigureCanvasTkAgg(f, master=surface_inverse_map_window)
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=2)
	ax = f.add_subplot(111, projection='3d')
	ax.margins(0,tight=True)
	ax.set_xlabel("X-axis ("+self.distance_unit+")")
	ax.set_ylabel("Y-axis ("+self.distance_unit+")")
	ax.set_zlabel("1/$\overline{\lambda}_{Local}$ ("+self.distance_unit+"$^{-1}$)")
	ax.zaxis.labelpad = 15
	ax.yaxis.labelpad = 15
	ax.xaxis.labelpad = 15
	sp_N = np.array([len(s) for s in self.analysis_data.spacings])
	sp_6_mean = np.mean(np.concatenate(self.analysis_data.spacings[np.where(sp_N == 6)[0]]))
	x,y,_,z = pdas_strain.copy().T
	z = z*sp_6_mean+sp_6_mean
	z = 1/z
	z_loc = (np.isnan(z) == False)&(np.isinf(z) == False)
	x=x[z_loc]
	y=y[z_loc]
	z=z[z_loc]
	try:
		a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
	except:
		try:
			a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
		except:
			print("Linalg failed. Falling back to minimize.")
			a_x,b_x,a_y,b_y,c = minimize(lambda c: np.sum((z - (c[0]*(x)**2+c[1]*(x)+c[2]*(y)**2+c[3]*(y)+c[4])).reshape(-1)**2),x0 = [0,0,0,0,0],method="Nelder-Mead",tol=1e-10  ).x
	truth_grid = np.zeros(self.main_image.shape[:2])
	cv2.fillPoly(truth_grid,[self.analysis_data.boundary_coords],1)
	if self.analysis_data.porosity:
		if len(self.analysis_data.sample_inner_bounds) > 0:
			cv2.drawContours(truth_grid,self.analysis_data.sample_inner_bounds,-1,0,-1)
	truth_grid[truth_grid == 0] = np.nan
	x_loc,y_loc,w,h = cv2.boundingRect(truth_grid.astype(np.uint8))
	yy,xx= np.meshgrid(np.linspace(y_loc,y_loc+h-1,500,dtype=int),np.linspace(x_loc,x_loc+w-1,500,dtype=int))
	truth_grid = truth_grid[(yy.reshape(-1),xx.reshape(-1))].reshape(500,500)
	wire = a_x*xx**2+b_x*xx+a_y*yy**2+b_y*yy+c
	xx = xx*self.pixel_scaling[2]
	yy = yy*self.pixel_scaling[2]
	wire[np.isnan(wire)] = 0
	wire[np.isinf(wire)] = 0
	analysis_wire = wire[truth_grid==1]
	vmin = np.mean(analysis_wire)-2*np.std(analysis_wire)
	vmax = np.mean(analysis_wire)+2*np.std(analysis_wire)
	wire[truth_grid!=1] = np.nan
	surface_plot = ax.plot_surface(xx,yy,wire,cmap="jet_r",vmin=vmin,vmax=vmax)
	scatter_plot = None
	def swap_scatter():
		nonlocal scatter_plot
		if scatter_plot == None:
			scatter_plot = ax.scatter(x*self.pixel_scaling[2],y*self.pixel_scaling[2],z,c="black")
			f.canvas.draw()
		else:
			scatter_plot.remove()
			scatter_plot = None
			f.canvas.draw()

	t_10 = np.log10(np.mean(analysis_wire))
	t_10 = int(t_10  if t_10  >= 0 else t_10-1)
	low_bound = np.round(np.min(analysis_wire),2-t_10)
	low_bound -= low_bound%10**(t_10-1)
	high_bound = np.round(np.max(analysis_wire),2-t_10)
	high_bound += 10**(t_10-1)-high_bound%10**(t_10-1)
	low_tick_ind = np.where(low_bound-ax.get_zticks()>0)[0]
	low_tick_ind = low_tick_ind[-1] if len(low_tick_ind)>0 else 0
	high_tick_ind = np.where(high_bound-ax.get_zticks()<0)[0]
	high_tick_ind = high_tick_ind[0] if len(high_tick_ind)>0 else -1
	axis_range_low = np.round(ax.get_zticks()[low_tick_ind],-(t_10-1))
	axis_range_high = np.round(ax.get_zticks()[high_tick_ind],-(t_10-1))
	axis_scale = 1
	ax.set_zlim(axis_range_low,axis_range_high)
	def on_motion(e):
		nonlocal surface_plot, axis_scale
		new_ticks = surface_plot.axes.get_zticks()*axis_scale
		t_10 = np.log10(np.mean(np.diff(new_ticks)))
		t_10 = int(t_10  if t_10  >= 0 else t_10-1)
		surface_plot.axes.set_zticklabels(np.round(new_ticks,-(t_10-1)))
	f.canvas.mpl_connect('motion_notify_event', on_motion)
	def inverse_LPS_right_click(event):
		try:
			inverse_LPS_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()

	def inverse_show_scale():
		inverse_scale_window.transient(surface_inverse_map_window)
		inverse_scale_window.update()
		inverse_scale_window.deiconify()

	def inverse_show_range():
		range_low_text.set(np.float32(axis_range_low*axis_scale))
		range_high_text.set(np.float32(axis_range_high*axis_scale))
		inverse_range_window.transient(surface_inverse_map_window)
		inverse_range_window.update()
		inverse_range_window.deiconify()

	def inverse_set_scale():
		nonlocal surface_plot, axis_scale
		if len(scale_text.get()) > 0:
			try:
				axis_scale = float(scale_text.get())
				axis_scale = int(axis_scale) if axis_scale == int(axis_scale) else axis_scale
				ax.set_zlabel(str(axis_scale)+"/$\overline{\lambda}_{Local}$ ("+self.distance_unit+"$^{-1}$)")
				on_motion(None)
				f.canvas.draw()
			except:
				scale_text.set(str(axis_scale))
				axis_scale = int(axis_scale) if axis_scale == int(axis_scale) else axis_scale
				ax.set_zlabel(str(axis_scale)+"/$\overline{\lambda}_{Local}$ ("+self.distance_unit+"$^{-1}$)")
				on_motion(None)
				f.canvas.draw()
	def inverse_set_range():
		nonlocal axis_scale, ax, axis_range_low, axis_range_high
		if len(scale_text.get()) > 0:
			try:
				low_txt = float(range_low_text.get())/axis_scale
				high_txt = float(range_high_text.get())/axis_scale
				axis_range_low = low_txt
				axis_range_high = high_txt
				ax.set_zlim(axis_range_low,axis_range_high)
				on_motion(None)
				f.canvas.draw()
			except:
				range_low_text.set(axis_range_low*axis_scale)
				range_high_text.set(axis_range_high*axis_scale)
				on_motion(None)
				f.canvas.draw()
	def inverse_cancel():
		inverse_scale_close()
	def inverse_range_cancel():
		inverse_range_close()
	surface_inverse_map_window.bind("<Button-3>",inverse_LPS_right_click)
	inverse_LPS_popup = Menu(surface_inverse_map_window,tearoff=0)
	scatter_var = BooleanVar()
	inverse_LPS_popup.add_checkbutton(label="Scatter points",command=swap_scatter, offvalue=0, onvalue=1, variable=scatter_var)
	inverse_LPS_popup.add_command(label="Set Z Scaling factor",command=inverse_show_scale)
	inverse_LPS_popup.add_command(label="Set Z range",command=inverse_show_range)
	def only_numbers(char):
		return char.isdigit() or char=="."
	vcmd = (inverse_scale_window.register(only_numbers),'%S')
	scale_text = StringVar()
	scale_text.set(str(axis_scale))
	scale_text_box = Entry(inverse_scale_window,width=5,validate="key",validatecommand=vcmd, textvariable = scale_text)
	scale_text_box.grid(column=0,row=0,columnspan=2,pady=5,padx=20)
	scale_apply = Button(inverse_scale_window,text="Apply",command=inverse_set_scale)
	scale_apply.grid(column=1,row=1,padx=10,pady=5)
	scale_cancel = Button(inverse_scale_window,text="Cancel",command=inverse_cancel)
	scale_cancel.grid(column=0,row=1,padx=10,pady=5)
	on_motion(None)
	f.canvas.draw()
	surface_inverse_map_window.update()
	surface_inverse_map_window.deiconify()
	inverse_scale_window.update()
	inverse_scale_window.geometry(f"+{(self.screen_width-inverse_scale_window.winfo_width())//2}+{(self.screen_height-inverse_scale_window.winfo_height())//2}")
	range_low_text = StringVar()
	range_low_text.set(str(axis_range_low))
	range_high_text = StringVar()
	range_high_text.set(str(axis_range_high))
	range_text_box_low = Entry(inverse_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_low_text)
	range_text_box_low.grid(column=0,row=0,pady=5,padx=2)
	range_text_box_high = Entry(inverse_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_high_text)
	range_text_box_high.grid(column=2,row=0,pady=5,padx=2)
	Label(inverse_range_window,text="-",width=1,justify=CENTER).grid(column=1,row=0)
	range_apply = Button(inverse_range_window,text="Apply",command=inverse_set_range)
	range_apply.grid(column=2,row=1,padx=2,pady=5)
	range_cancel = Button(inverse_range_window,text="Cancel",command=inverse_range_cancel)
	range_cancel.grid(column=0,row=1,padx=2,pady=5)
	inverse_range_window.update()
	inverse_range_window.geometry(f"+{(self.screen_width-inverse_scale_window.winfo_width())//2}+{(self.screen_height-inverse_scale_window.winfo_height())//2}")
	f1 = Frame(surface_inverse_map_window)
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
	def export_inverse_map():
		img_save_name = filedialog.asksaveasfilename(title="Select file to export all graphs to",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			f.savefig(img_save_name,dpi=float(dpi.get()))
	f2 = Frame(surface_inverse_map_window)
	f2.grid(row=1, column=1, sticky=N+S+E+W)
	Button(f2,text="Export",command=export_inverse_map).pack(side=RIGHT,pady=5,padx=5)

def gui_surface_map(self):
	if not self.check_analysis_data():
		return
	
	if type(self.analysis_data.pdas_strain) == bool:
		pdas_strain = self.analysis_data.calc_pdas_strain()
	else:
		pdas_strain = self.analysis_data.pdas_strain
	surface_surface_map_window = Toplevel()
	surface_surface_map_window.withdraw()
	ax = False
	surface_surface_map_window.title("3D Solutally Unstable Curvature")
	surface_surface_map_window.transient(self.window)
	surface_surface_map_window.grab_set()
	surface_surface_map_window.resizable(0,0)
	surface_scale_window = Toplevel(surface_surface_map_window)
	surface_scale_window.withdraw()
	surface_range_window = Toplevel(surface_surface_map_window)
	surface_range_window.withdraw()
	def surface_scale_close():
		surface_scale_window.transient(self.window)
		surface_scale_window.withdraw()
	def surface_range_close():
		surface_range_window.transient(self.window)
		surface_range_window.withdraw()
	def surface_surface_map_close():
		surface_scale_window.destroy()
		surface_range_window.destroy()
		surface_surface_map_window.destroy()
		nonlocal ax
		del ax
		plt.close('all')
	surface_surface_map_window.protocol("WM_DELETE_WINDOW",surface_surface_map_close)
	surface_scale_window.protocol("WM_DELETE_WINDOW", surface_scale_close)
	surface_scale_window.title("Set Z scale")
	surface_scale_window.transient(self.window)
	surface_scale_window.resizable(0,0)
	surface_range_window.protocol("WM_DELETE_WINDOW", surface_range_close)
	surface_range_window.title("Set Z range")
	surface_range_window.transient(self.window)
	surface_range_window.resizable(0,0)
	f = plt.figure(figsize=(5,4))
	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	f.set_size_inches(im_w/f.dpi,im_h/f.dpi)
	canvas = FigureCanvasTkAgg(f, master=surface_surface_map_window)
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=2)
	ax = f.add_subplot(111, projection='3d')
	ax.margins(0,tight=True)
	ax.set_xlabel("X-axis ("+self.distance_unit+")")
	ax.set_ylabel("Y-axis ("+self.distance_unit+")")
	ax.set_zlabel("$\overline{\lambda}_{Local}$ ("+self.distance_unit+")")
	ax.zaxis.labelpad = 15
	ax.yaxis.labelpad = 15
	ax.xaxis.labelpad = 15
	sp_N = np.array([len(s) for s in self.analysis_data.spacings])
	sp_6_mean = np.mean(np.concatenate(self.analysis_data.spacings[np.where(sp_N == 6)[0]]))
	x,y,_,z = pdas_strain.copy().T
	z = z*sp_6_mean+sp_6_mean
	z_loc = (np.isnan(z) == False)&(np.isinf(z) == False)
	x=x[z_loc]
	y=y[z_loc]
	z=z[z_loc]
	try:
		a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
	except:
		try:
			a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
		except:
			print("Linalg failed. Falling back to minimize.")
			a_x,b_x,a_y,b_y,c = minimize(lambda c: np.sum((z - (c[0]*(x)**2+c[1]*(x)+c[2]*(y)**2+c[3]*(y)+c[4])).reshape(-1)**2),x0 = [0,0,0,0,0],method="Nelder-Mead",tol=1e-10  ).x
	truth_grid = np.zeros(self.main_image.shape[:2])
	cv2.fillPoly(truth_grid,[self.analysis_data.boundary_coords],1)
	if self.analysis_data.porosity:
		if len(self.analysis_data.sample_inner_bounds) > 0:
			cv2.drawContours(truth_grid,self.analysis_data.sample_inner_bounds,-1,0,-1)
	truth_grid[truth_grid == 0] = np.nan
	x_loc,y_loc,w,h = cv2.boundingRect(truth_grid.astype(np.uint8))
	xx,yy = np.meshgrid(np.linspace(x_loc,x_loc+w-1,500,dtype=int),np.linspace(y_loc,y_loc+h-1,500,dtype=int))
	truth_grid = truth_grid[(yy.reshape(-1),xx.reshape(-1))].reshape(500,500)
	wire = a_x*xx**2+b_x*xx+a_y*yy**2+b_y*yy+c
	xx = xx*self.pixel_scaling[2]
	yy = yy*self.pixel_scaling[2]
	wire[np.isnan(wire)] = 0
	wire[np.isinf(wire)] = 0
	analysis_wire = wire[truth_grid==1]
	vmin = np.mean(analysis_wire)-2*np.std(analysis_wire)
	vmax = np.mean(analysis_wire)+2*np.std(analysis_wire)
	wire[truth_grid!=1] = np.nan
	surface_plot = ax.plot_surface(xx,yy,wire,cmap="jet",vmin=vmin,vmax=vmax)
	scatter_plot = None
	def swap_scatter():
		nonlocal scatter_plot
		if scatter_plot == None:
			scatter_plot = ax.scatter(x*self.pixel_scaling[2],y*self.pixel_scaling[2],z,c="black")
			f.canvas.draw()
		else:
			scatter_plot.remove()
			scatter_plot = None
			f.canvas.draw()

	t_10 = np.log10(np.mean(analysis_wire))
	t_10 = int(t_10  if t_10  >= 0 else t_10-1)
	low_bound = np.round(np.min(analysis_wire),2-t_10)
	low_bound -= low_bound%10**(t_10-1)
	high_bound = np.round(np.max(analysis_wire),2-t_10)
	high_bound += 10**(t_10-1)-high_bound%10**(t_10-1)
	low_tick_ind = np.where(low_bound-ax.get_zticks()>0)[0]
	low_tick_ind = low_tick_ind[-1] if len(low_tick_ind)>0 else 0
	high_tick_ind = np.where(high_bound-ax.get_zticks()<0)[0]
	high_tick_ind = high_tick_ind[0] if len(high_tick_ind)>0 else -1
	axis_range_low = np.round(ax.get_zticks()[low_tick_ind],-(t_10-1))
	axis_range_high = np.round(ax.get_zticks()[high_tick_ind],-(t_10-1))
	axis_scale = 1
	ax.set_zlim(axis_range_low,axis_range_high)

	def on_motion(e):
		nonlocal surface_plot, axis_scale
		new_ticks = surface_plot.axes.get_zticks()*axis_scale
		t_10 = np.log10(np.mean(np.diff(new_ticks)))
		t_10 = int(t_10  if t_10  >= 0 else t_10-1)
		surface_plot.axes.set_zticklabels(np.round(new_ticks,-(t_10-1)))
	f.canvas.mpl_connect('motion_notify_event', on_motion)
	def surface_LPS_right_click(event):
		try:
			surface_LPS_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()

	def surface_show_scale():
		surface_scale_window.transient(surface_surface_map_window)
		surface_scale_window.update()
		surface_scale_window.deiconify()

	def surface_show_range():
		range_low_text.set(np.float32(axis_range_low*axis_scale))
		range_high_text.set(np.float32(axis_range_high*axis_scale))
		surface_range_window.transient(surface_surface_map_window)
		surface_range_window.update()
		surface_range_window.deiconify()

	def surface_set_scale():
		nonlocal surface_plot, axis_scale
		if len(scale_text.get()) > 0:
			try:
				axis_scale = float(scale_text.get())
				axis_scale = int(axis_scale) if axis_scale == int(axis_scale) else axis_scale
				if axis_scale == 1:
					ax.set_zlabel("$\overline{\lambda}_{Local}$ ("+self.distance_unit+")")
				else:
					ax.set_zlabel(str(axis_scale)+r"$\times\overline{\lambda}_{Local}$ ("+self.distance_unit+")")
				on_motion(None)
				f.canvas.draw()
			except:
				scale_text.set(str(axis_scale))
				axis_scale = int(axis_scale) if axis_scale == int(axis_scale) else axis_scale
				if axis_scale == 1:
					ax.set_zlabel("$\overline{\lambda}_{Local}$ ("+self.distance_unit+"$^{-1}$)")
				else:
					ax.set_zlabel(str(axis_scale)+r"$\times\overline{\lambda}_{Local}$ ("+self.distance_unit+")")
				on_motion(None)
				f.canvas.draw()
	def surface_set_range():
		nonlocal axis_scale, ax, axis_range_low, axis_range_high
		if len(scale_text.get()) > 0:
			try:
				low_txt = float(range_low_text.get())/axis_scale
				high_txt = float(range_high_text.get())/axis_scale
				axis_range_low = low_txt
				axis_range_high = high_txt
				ax.set_zlim(axis_range_low,axis_range_high)
				on_motion(None)
				f.canvas.draw()
			except:
				range_low_text.set(axis_range_low*axis_scale)
				range_high_text.set(axis_range_high*axis_scale)
				on_motion(None)
				f.canvas.draw()
	def surface_cancel():
		surface_scale_close()
	def surface_range_cancel():
		surface_range_close()
	surface_surface_map_window.bind("<Button-3>",surface_LPS_right_click)
	surface_LPS_popup = Menu(surface_surface_map_window,tearoff=0)
	scatter_var = BooleanVar()
	surface_LPS_popup.add_checkbutton(label="Scatter points",command=swap_scatter, offvalue=0, onvalue=1, variable=scatter_var)
	surface_LPS_popup.add_command(label="Set Z Scaling factor",command=surface_show_scale)
	surface_LPS_popup.add_command(label="Set Z range",command=surface_show_range)
	def only_numbers(char):
		return char.isdigit() or char=="."
	vcmd = (surface_scale_window.register(only_numbers),'%S')
	scale_text = StringVar()
	scale_text.set(str(axis_scale))
	scale_text_box = Entry(surface_scale_window,width=5,validate="key",validatecommand=vcmd, textvariable = scale_text)
	scale_text_box.grid(column=0,row=0,columnspan=2,pady=5,padx=20)
	scale_apply = Button(surface_scale_window,text="Apply",command=surface_set_scale)
	scale_apply.grid(column=1,row=1,padx=10,pady=5)
	scale_cancel = Button(surface_scale_window,text="Cancel",command=surface_cancel)
	scale_cancel.grid(column=0,row=1,padx=10,pady=5)
	on_motion(None)
	f.canvas.draw()
	surface_surface_map_window.update()
	surface_surface_map_window.deiconify()
	surface_scale_window.update()
	surface_scale_window.geometry(f"+{(self.screen_width-surface_scale_window.winfo_width())//2}+{(self.screen_height-surface_scale_window.winfo_height())//2}")
	range_low_text = StringVar()
	range_low_text.set(str(axis_range_low))
	range_high_text = StringVar()
	range_high_text.set(str(axis_range_high))
	range_text_box_low = Entry(surface_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_low_text)
	range_text_box_low.grid(column=0,row=0,pady=5,padx=2)
	range_text_box_high = Entry(surface_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_high_text)
	range_text_box_high.grid(column=2,row=0,pady=5,padx=2)
	Label(surface_range_window,text="-",width=1,justify=CENTER).grid(column=1,row=0)
	range_apply = Button(surface_range_window,text="Apply",command=surface_set_range)
	range_apply.grid(column=2,row=1,padx=2,pady=5)
	range_cancel = Button(surface_range_window,text="Cancel",command=surface_range_cancel)
	range_cancel.grid(column=0,row=1,padx=2,pady=5)
	surface_range_window.update()
	surface_range_window.geometry(f"+{(self.screen_width-surface_scale_window.winfo_width())//2}+{(self.screen_height-surface_scale_window.winfo_height())//2}")
	f1 = Frame(surface_surface_map_window)
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
	def export_surface_map():
		img_save_name = filedialog.asksaveasfilename(title="Select file to export all graphs to",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			f.savefig(img_save_name,dpi=float(dpi.get()))
	f2 = Frame(surface_surface_map_window)
	f2.grid(row=1, column=1, sticky=N+S+E+W)
	Button(f2,text="Export",command=export_surface_map).pack(side=RIGHT,pady=5,padx=5)

def gui_inverse_2d_surface_map(self):
	if not self.check_analysis_data():
		return
	
	if type(self.analysis_data.pdas_strain) == bool:
		pdas_strain = self.analysis_data.calc_pdas_strain()
	else:
		pdas_strain = self.analysis_data.pdas_strain
	surface_inverse_map_window = Toplevel()
	surface_inverse_map_window.withdraw()
	ax = False
	surface_inverse_map_window.title("2D Solutally Stable Curvature")
	surface_inverse_map_window.transient(self.window)
	surface_inverse_map_window.grab_set()
	surface_inverse_map_window.resizable(0,0)
	inverse_range_window = Toplevel(surface_inverse_map_window)
	inverse_range_window.withdraw()
	def inverse_range_close():
		inverse_range_window.transient(self.window)
		inverse_range_window.withdraw()
	def surface_inverse_map_close():
		inverse_range_window.destroy()
		surface_inverse_map_window.destroy()
	surface_inverse_map_window.protocol("WM_DELETE_WINDOW",surface_inverse_map_close)
	inverse_range_window.protocol("WM_DELETE_WINDOW", inverse_range_close)
	inverse_range_window.title("Set colour range")
	inverse_range_window.transient(self.window)
	inverse_range_window.resizable(0,0)

	sp_N = np.array([len(s) for s in self.analysis_data.spacings])
	sp_6_mean = np.mean(np.concatenate(self.analysis_data.spacings[np.where(sp_N == 6)[0]]))
	x,y,_,z = pdas_strain.copy().T
	z = z*sp_6_mean+sp_6_mean
	z = 1/z
	z_loc = (np.isnan(z) == False)&(np.isinf(z) == False)
	x=x[z_loc]
	y=y[z_loc]
	z=z[z_loc]
	try:
		a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
	except:
		try:
			a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
		except:
			print("Linalg failed. Falling back to minimize.")
			a_x,b_x,a_y,b_y,c = minimize(lambda c: np.sum((z - (c[0]*(x)**2+c[1]*(x)+c[2]*(y)**2+c[3]*(y)+c[4])).reshape(-1)**2),x0 = [0,0,0,0,0],method="Nelder-Mead",tol=1e-10  ).x
	truth_grid = np.zeros(self.main_image.shape[:2])
	cv2.fillPoly(truth_grid,[self.analysis_data.boundary_coords],1)
	if self.analysis_data.porosity:
		if len(self.analysis_data.sample_inner_bounds) > 0:
			cv2.drawContours(truth_grid,self.analysis_data.sample_inner_bounds,-1,0,-1)

	truth_grid = cv2.resize(truth_grid,(truth_grid.shape[1]//2,truth_grid.shape[0]//2),interpolation = cv2.INTER_CUBIC)
	yy,xx = np.where(truth_grid==1)
	yy *= 2
	xx *= 2
	wire = a_x*xx**2+b_x*xx+a_y*yy**2+b_y*yy+c
	wire[np.isnan(wire)] = 0
	wire[np.isinf(wire)] = 0
	wire_std = np.float32(np.std(wire))
	wire_mean = np.float32(np.mean(wire))
	vmin = wire_mean-2*wire_std
	vmax = wire_mean+2*wire_std
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap="jet")
	colour_bar = denmap_stats.create_colourbar(truth_grid,vmin,vmax)
	colour_wire = np.round(mapper.to_rgba(wire)[:,:3]*255).astype(np.uint8)
	colour_map = np.zeros(truth_grid.shape[:2]+(3,),np.uint8)
	truth_grid_loc = np.where(truth_grid==1)
	colour_map[truth_grid_loc] = colour_wire
	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	canvas = Canvas(surface_inverse_map_window,bd=-2)
	canvas.pack(fill=BOTH, expand=YES)
	surface_inverse_map_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
	self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
	canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
	def LPS_right_click(event):
		try:
			LPS_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def swap_scatter():
		nonlocal colour_map
		if scatter_var.get():
			for x,y in self.analysis_data.points.astype(int):
				#cv2.drawMarker(final_image,(x,y),color=(0,0,0),markerType=cv2.MARKER_CROSS,markerSize = max_mS,thickness=maxThick)
				#cv2.drawMarker(final_image,(x,y),color=(255,255,255),markerType=cv2.MARKER_CROSS,markerSize = min_mS,thickness=minThick)
				cv2.circle(colour_map,(x//2,y//2),radius = max_mS,color=(0,0,0),thickness=-1)
				cv2.circle(colour_map,(x//2,y//2),radius = min_mS,color=(255,255,255),thickness=-1)
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.update()
		else:
			if scatter_cbar_var.get():
				colour_map = np.zeros((truth_grid.shape[0],truth_grid.shape[1]+colour_bar.shape[1],3),np.uint8)
				colour_map[truth_grid_loc] = colour_wire
				colour_map[:,truth_grid.shape[1]:] = colour_bar[:truth_grid.shape[0],:]
			else:
				colour_map = np.zeros(truth_grid.shape[:2]+(3,),np.uint8)
				colour_map[truth_grid_loc] = colour_wire
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.update()
	def swap_cbar():
		nonlocal colour_map, im_w, im_h, scale
		if scatter_cbar_var.get():
			scale = min(self.screen_width/(truth_grid.shape[1]+colour_bar.shape[1]),self.screen_height/truth_grid.shape[0])*0.9
			im_h = int(colour_map.shape[0]*scale)
			im_w = int((truth_grid.shape[1]+colour_bar.shape[1])*scale)
			colour_map_w_bar = np.zeros((truth_grid.shape[0],truth_grid.shape[1]+colour_bar.shape[1],3),np.uint8)
			colour_map_w_bar[:truth_grid.shape[0],:truth_grid.shape[1]] = colour_map
			colour_map_w_bar[:,truth_grid.shape[1]:] = colour_bar[:truth_grid.shape[0],:]
			colour_map = colour_map_w_bar
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.geometry(str(im_w)+"x"+str(im_h)+"+"+str(int((self.screen_width-im_w)//2))+"+"+str(self.window_y_pos))
			surface_inverse_map_window.update()
		else:
			scale = min(self.screen_width/truth_grid.shape[1],self.screen_height/truth_grid.shape[0])*0.9
			im_h = int(truth_grid.shape[0]*scale)
			im_w = int(truth_grid.shape[1]*scale)
			colour_map = colour_map[:truth_grid.shape[0],:truth_grid.shape[1]]
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.geometry(str(im_w)+"x"+str(im_h)+"+"+str(int((self.screen_width-im_w)//2))+"+"+str(self.window_y_pos))
			surface_inverse_map_window.update()
	def surface_save():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save map to",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			save_w = self.main_image.shape[1]
			if scatter_cbar_var.get():
				save_w += colour_bar.shape[1]*2
			cv2.imwrite(img_save_name,cv2.resize(colour_map,(save_w,self.main_image.shape[0]),interpolation = cv2.INTER_CUBIC)[:,:,::-1])
	def inverse_show_range():
		std_var.set(std_current)
		range_low_text.set(np.float32(vmin))
		range_high_text.set(np.float32(vmax))
		inverse_range_window.transient(surface_inverse_map_window)
		inverse_range_window.update()
		inverse_range_window.deiconify()
	surface_inverse_map_window.bind("<Button-3>",LPS_right_click)
	LPS_popup = Menu(surface_inverse_map_window,tearoff=0)
	scatter_var = BooleanVar()
	scatter_cbar_var = BooleanVar()
	LPS_popup.add_checkbutton(label="Scatter points",command=swap_scatter, offvalue=0, onvalue=1, variable=scatter_var)
	LPS_popup.add_checkbutton(label="Colour bar",command=swap_cbar, offvalue=0, onvalue=1, variable=scatter_cbar_var)
	LPS_popup.add_command(label="Set Colour Range",command=inverse_show_range)
	LPS_popup.add_command(label="Save Image",command=surface_save)
	if type(self.NCC_temp) != bool:
		rad = (self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/10
	else:
		if type(self.FFT_in) == bool:
			if len(self.original_image.shape) < 3:
				self.FFT_in = wg.calculate_r_in(self.original_image)
			else:
				self.FFT_in = wg.calculate_r_in(self.original_image[:,:,0])
			rad = self.FFT_in
		else:
			rad = self.FFT_in
	min_mS = np.round(rad).astype(int)
	max_mS = np.round(1.4*rad).astype(int)
	#Colour Range
	std_current = 0
	def only_numbers(char):
		return char.isdigit() or char=="."
	vcmd = (inverse_range_window.register(only_numbers),'%S')
	def inverse_set_range():
		nonlocal vmin, vmax, norm, mapper, colour_bar, colour_map, colour_wire, im_w, im_h, scale, std_current
		vmin = float(range_low_text.get())
		vmax = float(range_high_text.get())
		if std_var.get():
			vmin = vmin*wire_std+wire_mean
			vmax = vmax*wire_std+wire_mean
			colour_bar = denmap_stats.create_colourbar_std(truth_grid,wire_std,wire_mean,vmin,vmax)
			std_current = 1
		else:
			colour_bar = denmap_stats.create_colourbar(truth_grid,vmin,vmax)
			std_current = 0
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
		mapper = cm.ScalarMappable(norm=norm, cmap="jet")
		colour_wire = np.round(mapper.to_rgba(wire)[:,:3]*255).astype(np.uint8)
		if scatter_cbar_var.get():
			colour_map = np.zeros((truth_grid.shape[0],truth_grid.shape[1]+colour_bar.shape[1],3),np.uint8)
			colour_map[:truth_grid.shape[0],truth_grid.shape[1]:] = colour_bar[:truth_grid.shape[0],:]
			colour_map[truth_grid_loc] = colour_wire
		else:
			colour_map = np.zeros((truth_grid.shape[0],truth_grid.shape[1],3),np.uint8)
			colour_map[truth_grid_loc] = colour_wire
		scale = min(self.screen_width/(truth_grid.shape[1]+colour_bar.shape[1]),self.screen_height/truth_grid.shape[0])*0.9
		im_h = int(colour_map.shape[0]*scale)
		im_w = int(colour_map.shape[1]*scale)
		self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
		canvas.delete("all")
		canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
		surface_inverse_map_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
		surface_inverse_map_window.update()
		#inverse_range_close()
	def std_checkbox():
		if std_var.get():
			range_low_text.set(str(np.round((float(range_low_text.get())-wire_mean)/wire_std,5)))
			range_high_text.set(str(np.round((float(range_high_text.get())-wire_mean)/wire_std,5)))
		else:
			range_low_text.set(str(float(range_low_text.get())*wire_std+wire_mean))
			range_high_text.set(str(float(range_high_text.get())*wire_std+wire_mean))
	range_low_text = StringVar()
	range_low_text.set(str(np.float32(vmin)))
	range_high_text = StringVar()
	range_high_text.set(str(np.float32(vmax)))
	range_text_box_low = Entry(inverse_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_low_text)
	range_text_box_low.grid(column=0,row=0,pady=5,padx=2)
	range_text_box_high = Entry(inverse_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_high_text)
	range_text_box_high.grid(column=2,row=0,pady=5,padx=2)
	Label(inverse_range_window,text="-",width=1,justify=CENTER).grid(column=1,row=0)
	std_var = IntVar()
	Checkbutton(inverse_range_window, text="St. Dev.", variable=std_var, command=std_checkbox).grid(row=1,column=0, sticky=W)
	range_apply = Button(inverse_range_window,text="Apply",command=inverse_set_range)
	range_apply.grid(column=2,row=2,padx=2,pady=5)
	range_cancel = Button(inverse_range_window,text="Cancel",command=inverse_range_close)
	range_cancel.grid(column=0,row=2,padx=2,pady=5)
	inverse_range_window.update()
	inverse_range_window.geometry(f"+{(self.screen_width-inverse_range_window.winfo_width())//2}+{(self.screen_height-inverse_range_window.winfo_height())//2}")
	surface_inverse_map_window.update()
	surface_inverse_map_window.deiconify()

def gui_2d_surface_map(self):
	if not self.check_analysis_data():
		return
	
	#if type(analysis_data.nn_interpolated_values) == bool:
	#	nn_interpolated_values,truth_grid = analysis_data.calc_heat_map(main_image)
	#else:
	#	nn_interpolated_values = analysis_data.nn_interpolated_values
	#	truth_grid = analysis_data.nn_truth_grid
	if type(self.analysis_data.pdas_strain) == bool:
		pdas_strain = self.analysis_data.calc_pdas_strain()
	else:
		pdas_strain = self.analysis_data.pdas_strain
	surface_inverse_map_window = Toplevel()
	surface_inverse_map_window.withdraw()
	ax = False
	surface_inverse_map_window.title("2D Solutally Unstable Curvature")
	surface_inverse_map_window.transient(self.window)
	surface_inverse_map_window.grab_set()
	surface_inverse_map_window.resizable(0,0)
	inverse_range_window = Toplevel(surface_inverse_map_window)
	inverse_range_window.withdraw()
	def inverse_range_close():
		inverse_range_window.transient(self.window)
		inverse_range_window.withdraw()
	def surface_inverse_map_close():
		inverse_range_window.destroy()
		surface_inverse_map_window.destroy()
	surface_inverse_map_window.protocol("WM_DELETE_WINDOW",surface_inverse_map_close)
	inverse_range_window.protocol("WM_DELETE_WINDOW", inverse_range_close)
	inverse_range_window.title("Set colour range")
	inverse_range_window.transient(self.window)
	inverse_range_window.resizable(0,0)

	sp_N = np.array([len(s) for s in self.analysis_data.spacings])
	sp_6_mean = np.mean(np.concatenate(self.analysis_data.spacings[np.where(sp_N == 6)[0]]))
	x,y,_,z = pdas_strain.copy().T
	z = z*sp_6_mean+sp_6_mean
	try:
		a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
	except:
		try:
			a_x,a_y,b_x,b_y,c = quadratic_solv(x,y,z)
		except:
			print("Linalg failed. Falling back to minimize.")
			a_x,b_x,a_y,b_y,c = minimize(lambda c: np.sum((z - (c[0]*(x)**2+c[1]*(x)+c[2]*(y)**2+c[3]*(y)+c[4])).reshape(-1)**2),x0 = [0,0,0,0,0],method="Nelder-Mead",tol=1e-10  ).x
	truth_grid = np.zeros(self.main_image.shape[:2])
	cv2.fillPoly(truth_grid,[self.analysis_data.boundary_coords],1)
	if self.analysis_data.porosity:
		if len(self.analysis_data.sample_inner_bounds) > 0:
			cv2.drawContours(truth_grid,self.analysis_data.sample_inner_bounds,-1,0,-1)

	truth_grid = cv2.resize(truth_grid,(truth_grid.shape[1]//2,truth_grid.shape[0]//2),interpolation = cv2.INTER_CUBIC)
	yy,xx = np.where(truth_grid==1)
	yy *= 2
	xx *= 2
	wire = a_x*xx**2+b_x*xx+a_y*yy**2+b_y*yy+c
	z_theory = a_x*x**2+b_x*x+a_y*y**2+b_y*y+c
	error = np.mean(abs(z-a_x*x**2+b_x*x+a_y*y**2+b_y*y+c))
	error_std = np.std(abs(z-a_x*x**2+b_x*x+a_y*y**2+b_y*y+c))
	wire[np.isnan(wire)] = 0
	wire[np.isinf(wire)] = 0
	wire_std = np.float32(np.std(wire))
	wire_mean = np.float32(np.mean(wire))
	vmin = wire_mean-2*wire_std
	vmax = wire_mean+2*wire_std
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap="jet")
	colour_bar = denmap_stats.create_colourbar(truth_grid,vmin,vmax)
	colour_wire = np.round(mapper.to_rgba(wire)[:,:3]*255).astype(np.uint8)
	colour_map = np.zeros(truth_grid.shape[:2]+(3,),np.uint8)
	truth_grid_loc = np.where(truth_grid==1)
	colour_map[truth_grid_loc] = colour_wire

	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	canvas = Canvas(surface_inverse_map_window,bd=-2)

	canvas.pack(fill=BOTH, expand=YES)
	surface_inverse_map_window.geometry(str(im_w)+"x"+str(im_h)+"+"+str(int((self.screen_width-im_w)//2))+"+"+str(self.window_y_pos))
	self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
	canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
	def LPS_right_click(event):
		try:
			LPS_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def swap_scatter():
		nonlocal colour_map
		if scatter_var.get():
			for x,y in self.analysis_data.points.astype(int):
				#cv2.drawMarker(final_image,(x,y),color=(0,0,0),markerType=cv2.MARKER_CROSS,markerSize = max_mS,thickness=maxThick)
				#cv2.drawMarker(final_image,(x,y),color=(255,255,255),markerType=cv2.MARKER_CROSS,markerSize = min_mS,thickness=minThick)
				cv2.circle(colour_map,(x//2,y//2),radius = max_mS,color=(0,0,0),thickness=-1)
				cv2.circle(colour_map,(x//2,y//2),radius = min_mS,color=(255,255,255),thickness=-1)
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.update()
		else:
			if scatter_cbar_var.get():
				colour_map = np.zeros((truth_grid.shape[0],truth_grid.shape[1]+colour_bar.shape[1],3),np.uint8)
				colour_map[truth_grid_loc] = colour_wire
				colour_map[:,truth_grid.shape[1]:] = colour_bar[:truth_grid.shape[0],:]
			else:
				colour_map = np.zeros(truth_grid.shape[:2]+(3,),np.uint8)
				colour_map[truth_grid_loc] = colour_wire
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.update()
	def swap_cbar():
		nonlocal colour_map, im_w, im_h, scale
		if scatter_cbar_var.get():
			scale = min(self.screen_width/(truth_grid.shape[1]+colour_bar.shape[1]),self.screen_height/truth_grid.shape[0])*0.9
			im_h = int(colour_map.shape[0]*scale)
			im_w = int((truth_grid.shape[1]+colour_bar.shape[1])*scale)
			colour_map_w_bar = np.zeros((truth_grid.shape[0],truth_grid.shape[1]+colour_bar.shape[1],3),np.uint8)
			colour_map_w_bar[:truth_grid.shape[0],:truth_grid.shape[1]] = colour_map
			colour_map_w_bar[:,truth_grid.shape[1]:] = colour_bar[:truth_grid.shape[0],:]
			colour_map = colour_map_w_bar
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.geometry(str(im_w)+"x"+str(im_h)+"+"+str(int((self.screen_width-im_w)//2))+"+"+str(self.window_y_pos))
			surface_inverse_map_window.update()
		else:
			scale = min(self.screen_width/truth_grid.shape[1],self.screen_height/truth_grid.shape[0])*0.9
			im_h = int(truth_grid.shape[0]*scale)
			im_w = int(truth_grid.shape[1]*scale)
			colour_map = colour_map[:truth_grid.shape[0],:truth_grid.shape[1]]
			self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			canvas.delete("all")
			canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
			surface_inverse_map_window.geometry(str(im_w)+"x"+str(im_h)+"+"+str(int((self.screen_width-im_w)//2))+"+"+str(self.window_y_pos))
			surface_inverse_map_window.update()
	def surface_save():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save map to",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			save_w = self.main_image.shape[1]
			if scatter_cbar_var.get():
				save_w += colour_bar.shape[1]*2
			cv2.imwrite(img_save_name,cv2.resize(colour_map,(save_w,self.main_image.shape[0]),interpolation = cv2.INTER_CUBIC)[:,:,::-1])
	def inverse_show_range():
		std_var.set(std_current)
		range_low_text.set(np.float32(vmin))
		range_high_text.set(np.float32(vmax))
		inverse_range_window.transient(surface_inverse_map_window)
		inverse_range_window.update()
		inverse_range_window.deiconify()
	surface_inverse_map_window.bind("<Button-3>",LPS_right_click)
	LPS_popup = Menu(surface_inverse_map_window,tearoff=0)
	scatter_var = BooleanVar()
	scatter_cbar_var = BooleanVar()
	LPS_popup.add_checkbutton(label="Scatter points",command=swap_scatter, offvalue=0, onvalue=1, variable=scatter_var)
	LPS_popup.add_checkbutton(label="Colour bar",command=swap_cbar, offvalue=0, onvalue=1, variable=scatter_cbar_var)
	LPS_popup.add_command(label="Set Colour Range",command=inverse_show_range)
	LPS_popup.add_command(label="Save Image",command=surface_save)
	if type(self.NCC_temp) != bool:
		rad = (self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/10
	else:
		if type(self.FFT_in) == bool:
			if len(self.original_image.shape) < 3:
				self.FFT_in = wg.calculate_r_in(self.original_image)
			else:
				self.FFT_in = wg.calculate_r_in(self.original_image[:,:,0])
			rad = self.FFT_in
		else:
			rad = self.FFT_in
	min_mS = np.round(rad).astype(int)
	max_mS = np.round(1.4*rad).astype(int)
	#Colour Range
	std_current = 0
	def only_numbers(char):
		return char.isdigit() or char=="."
	vcmd = (inverse_range_window.register(only_numbers),'%S')
	def inverse_set_range():
		nonlocal vmin, vmax, norm, mapper, colour_bar, colour_map, colour_wire, im_w, im_h, scale, std_current
		vmin = float(range_low_text.get())
		vmax = float(range_high_text.get())
		if std_var.get():
			vmin = vmin*wire_std+wire_mean
			vmax = vmax*wire_std+wire_mean
			colour_bar = denmap_stats.create_colourbar_std(truth_grid,wire_std,wire_mean,vmin,vmax)
			std_current = 1
		else:
			colour_bar = denmap_stats.create_colourbar(truth_grid,vmin,vmax)
			std_current = 0
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
		mapper = cm.ScalarMappable(norm=norm, cmap="jet")
		colour_wire = np.round(mapper.to_rgba(wire)[:,:3]*255).astype(np.uint8)
		if scatter_cbar_var.get():
			colour_map = np.zeros((truth_grid.shape[0],truth_grid.shape[1]+colour_bar.shape[1],3),np.uint8)
			colour_map[:truth_grid.shape[0],truth_grid.shape[1]:] = colour_bar[:truth_grid.shape[0],:]
			colour_map[truth_grid_loc] = colour_wire
		else:
			colour_map = np.zeros((truth_grid.shape[0],truth_grid.shape[1],3),np.uint8)
			colour_map[truth_grid_loc] = colour_wire
		scale = min(self.screen_width/(truth_grid.shape[1]+colour_bar.shape[1]),self.screen_height/truth_grid.shape[0])*0.9
		im_h = int(colour_map.shape[0]*scale)
		im_w = int(colour_map.shape[1]*scale)
		self.fit_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(colour_map,(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
		canvas.delete("all")
		canvas.create_image(0,0,anchor=NW,image=self.fit_display_image)
		surface_inverse_map_window.geometry(f"{im_w}x{im_h}"+str()+"+"+str(int((self.screen_width-im_w)//2))+"+"+str(self.window_y_pos))
		surface_inverse_map_window.update()
		#inverse_range_close()
	def std_checkbox():
		if std_var.get():
			range_low_text.set(str(np.round((float(range_low_text.get())-wire_mean)/wire_std,5)))
			range_high_text.set(str(np.round((float(range_high_text.get())-wire_mean)/wire_std,5)))
		else:
			range_low_text.set(str(float(range_low_text.get())*wire_std+wire_mean))
			range_high_text.set(str(float(range_high_text.get())*wire_std+wire_mean))
	range_low_text = StringVar()
	range_low_text.set(str(np.float32(vmin)))
	range_high_text = StringVar()
	range_high_text.set(str(np.float32(vmax)))
	range_text_box_low = Entry(inverse_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_low_text)
	range_text_box_low.grid(column=0,row=0,pady=5,padx=2)
	range_text_box_high = Entry(inverse_range_window,width=8,validate="key",validatecommand=vcmd, textvariable = range_high_text)
	range_text_box_high.grid(column=2,row=0,pady=5,padx=2)
	Label(inverse_range_window,text="-",width=1,justify=CENTER).grid(column=1,row=0)
	std_var = IntVar()
	Checkbutton(inverse_range_window, text="St. Dev.", variable=std_var, command=std_checkbox).grid(row=1,column=0, sticky=W)
	range_apply = Button(inverse_range_window,text="Apply",command=inverse_set_range)
	range_apply.grid(column=2,row=2,padx=2,pady=5)
	range_cancel = Button(inverse_range_window,text="Cancel",command=inverse_range_close)
	range_cancel.grid(column=0,row=2,padx=2,pady=5)
	inverse_range_window.update()
	inverse_range_window.geometry(f"+{(self.screen_width-inverse_range_window.winfo_width())//2}+{(self.screen_height-inverse_range_window.winfo_height())//2}")
	surface_inverse_map_window.update()
	surface_inverse_map_window.deiconify()