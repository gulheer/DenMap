from tkinter import Toplevel, Canvas, Menu, BooleanVar, filedialog, NW, BOTH, YES
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import numpy as np
import denmap_src.width_gui as wg
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

def gui_calculate_display_voronoi(self):
	if not self.check_analysis_data():
		return
	voronoi_window = Toplevel()
	voronoi_window.withdraw()
	def voronoi_close():
		voronoi_window.destroy()
		if hasattr(self,'voronoi_display_image'):
			del self.voronoi_display_image
		plt.close('all')
	voronoi_window.protocol("WM_DELETE_WINDOW", voronoi_close)
	voronoi_window.title("Voronoi Overlay Image")
	voronoi_window.transient(self.window)
	voronoi_window.grab_set()
	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	voronoi_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
	voronoi_canvas = Canvas(voronoi_window,bd=-2)
	voronoi_canvas.pack(fill=BOTH, expand=YES)
	voronoi_image = self.analysis_data.generate_voronoi_image(self.main_image,show_points=False,color_bar=True)
	self.voronoi_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(voronoi_image[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
	voronoi_canvas.create_image(0,0,anchor=NW,image=self.voronoi_display_image)
	voronoi_window.update()
	color_bar = BooleanVar()
	voronoi_window.resizable(0,0)
	def voronoi_right_click(event):
		try:
			voronoi_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def voronoi_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the voronoi plot",filetypes=[("Portable Network Graphics",".png"),("Tagged Image File Format",".tiff")],defaultextension=".png")
		if img_save_name:
			if color_bar.get():
				cv2.imwrite(img_save_name,voronoi_image[:, :, ::-1])
			else:
				cv2.imwrite(img_save_name,voronoi_image[:self.main_image.shape[0],:self.main_image.shape[1], ::-1])
	def voronoi_color_bar():
		nonlocal voronoi_canvas, voronoi_window
		if color_bar.get():
			scale_cb = min(self.screen_width/voronoi_image.shape[1],self.screen_height/voronoi_image.shape[0])*0.9
			im_h_cb = int(voronoi_image.shape[0]*scale_cb)
			im_w_cb = int(voronoi_image.shape[1]*scale_cb)
			self.voronoi_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(voronoi_image,(im_w_cb,im_h_cb),interpolation = cv2.INTER_CUBIC)))
			voronoi_canvas.delete("all")
			voronoi_canvas.create_image(0,0,anchor=NW,image=self.voronoi_display_image)
			voronoi_window.geometry(f"{im_w_cb}x{im_h_cb}+{int((self.screen_width-im_w_cb)//2)}+{self.window_y_pos}")
			voronoi_window.update()
		else:
			self.voronoi_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(voronoi_image[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			voronoi_canvas.delete("all")
			voronoi_canvas.create_image(0,0,anchor=NW,image=self.voronoi_display_image)
			voronoi_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
			voronoi_window.update()
	voronoi_window.bind("<Button-3>",voronoi_right_click)
	voronoi_popup = Menu(voronoi_window,tearoff=0)
	voronoi_popup.add_command(label="Save plot",command=voronoi_save_plot)
	voronoi_popup.add_checkbutton(label="Color bar",command=voronoi_color_bar, offvalue=0, onvalue=1, variable=color_bar)
	voronoi_window.update()
	voronoi_window.deiconify()

def image_heat_map(self,img,nn_interpolated_values,truth_grid,show_points=True,color_bar=True):
	v_min = np.mean(nn_interpolated_values)-3*np.std(nn_interpolated_values)
	v_max = np.mean(nn_interpolated_values)+3*np.std(nn_interpolated_values)
	norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max, clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap="jet")
	if color_bar:
		a = np.array([mapper.get_clim()])
		f = plt.figure(figsize=(0.8,3.77))
		i = plt.imshow(a, cmap=mapper.cmap)
		plt.gca().set_visible(False)
		f.tight_layout(pad=0)
		cax = plt.axes([0.06, 0.021, 0.32, 0.959])
		new_dpi = img.shape[0]/3.77
		f.dpi = new_dpi
		c_b = plt.colorbar(cax=cax,ticks=np.linspace(v_min, v_max, 7))
		c_b.set_ticklabels(["\u03BC-3\u03C3", "\u03BC-2\u03C3", "\u03BC-\u03C3", "\u03BC","\u03BC+\u03C3", "\u03BC+2\u03C3", "\u03BC+3\u03C3" ])
		f.canvas.draw()
		data = np.array(f.canvas.renderer.buffer_rgba())[:,:,0:3]
		plt.close('all')
		image_overlay = np.ones((img.shape[0],img.shape[1]+data.shape[1],3),dtype=np.uint8)*255
		if len(img.shape) < 3:
			image_overlay[:img.shape[0],:img.shape[1],:] = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
		else:
			image_overlay[:img.shape[0],:img.shape[1],:] = img
		image_overlay[:min(data.shape[0],image_overlay.shape[0]),-data.shape[1]:,:] = data[:min(data.shape[0],image_overlay.shape[0]),:,:]
	else:
		image_overlay = img.copy()
		if len(image_overlay.shape) < 3:
			image_overlay = cv2.cvtColor(image_overlay,cv2.COLOR_GRAY2RGB)
	nn_overlay = image_overlay.copy()
	truth_pos = np.where(truth_grid == 1)
	nn_overlay[truth_pos] = np.round(mapper.to_rgba(nn_interpolated_values[truth_pos])[:,:3]*255).astype(np.uint8)
	final_image = cv2.addWeighted(image_overlay,0.5,nn_overlay,0.5,0)
	if show_points:
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
		for x,y in self.analysis_data.points.astype(int):
			cv2.circle(final_image,(x,y),radius = max_mS,color=(0,0,0),thickness=-1)
			cv2.circle(final_image,(x,y),radius = min_mS,color=(255,255,255),thickness=-1)
	return final_image

def gui_calculate_heat_map(self):
	if not self.check_analysis_data():
		return
	heat_window = Toplevel()
	heat_window.withdraw()
	def heat_close():
		heat_window.destroy()
		if getattr(self,'heat_display_image'):
			del self.heat_display_image
	heat_window.protocol("WM_DELETE_WINDOW", heat_close)
	heat_window.title("Local Primary Spacing Overlay Image")
	heat_window.transient(self.window)
	heat_window.grab_set()
	heat_window.resizable(0,0)
	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	if type(self.analysis_data.nn_interpolated_values) == bool:
		nn_interpolated_values,truth_grid = self.analysis_data.calc_heat_map(self.main_image)
	else:
		nn_interpolated_values = self.analysis_data.nn_interpolated_values
		truth_grid = self.analysis_data.nn_truth_grid
	#plot_heat_map(ax,self.main_image,nn_interpolated_values,truth_grid)
	
	heat_image = self.image_heat_map(self.main_image,nn_interpolated_values,truth_grid)
	heat_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
	heat_canvas = Canvas(heat_window,bd=-2)
	heat_canvas.pack(fill=BOTH, expand=YES)
	self.heat_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(heat_image[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
	heat_canvas.create_image(0,0,anchor=NW,image=self.heat_display_image)
	heat_window.update()
	color_bar = BooleanVar()
	def heat_right_click(event):
		try:
			heat_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def heat_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the heat plot",filetypes=[("Portable Network Graphics",".png"),("Tagged Image File Format",".tiff")],defaultextension=".png")
		if img_save_name:
			if color_bar.get():
				cv2.imwrite(img_save_name,heat_image[:, :, ::-1])
			else:
				cv2.imwrite(img_save_name,heat_image[:self.main_image.shape[0],:self.main_image.shape[1], ::-1])
	def heat_color_bar():
		nonlocal heat_canvas, heat_window
		if color_bar.get():
			scale_cb = min(self.screen_width/heat_image.shape[1],self.screen_height/heat_image.shape[0])*0.9
			im_h_cb = int(heat_image.shape[0]*scale_cb)
			im_w_cb = int(heat_image.shape[1]*scale_cb)
			self.heat_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(heat_image,(im_w_cb,im_h_cb),interpolation = cv2.INTER_CUBIC)))
			heat_canvas.delete("all")
			heat_canvas.create_image(0,0,anchor=NW,image=self.heat_display_image)
			heat_window.geometry(f"{im_w_cb}x{im_h_cb}+{int((self.screen_width-im_w_cb)//2)}+{window_y_pos}")
			heat_window.update()
		else:
			self.heat_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(heat_image[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			heat_canvas.delete("all")
			heat_canvas.create_image(0,0,anchor=NW,image=self.heat_display_image)
			heat_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{window_y_pos}")
			heat_window.update()
	heat_window.bind("<Button-3>",heat_right_click)
	heat_popup = Menu(heat_window,tearoff=0)
	heat_popup.add_command(label="Save plot",command=heat_save_plot)
	heat_popup.add_checkbutton(label="Color bar",command=heat_color_bar, offvalue=0, onvalue=1, variable=color_bar)
	heat_window.update()
	heat_window.deiconify()