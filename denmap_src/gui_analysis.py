from tkinter import Toplevel, Button, Label, Menu, Entry, Scale, StringVar, messagebox, filedialog, LEFT, RIGHT, BOTH, W, E, HORIZONTAL
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import denmap_src.denmap_stats as denmap_stats
import pyperclip, cv2

def gui_measure_scale(self):
	measure_window = Toplevel()
	measure_window.withdraw()
	measure_window.title("Set Scale")
	measure_window.resizable(0,0)
	measure_window.transient(self.window)
	Label(measure_window, text="Pixel Distance:",justify=RIGHT).grid(row=0)
	Label(measure_window, text="Known Distance:",justify=RIGHT).grid(row=1)
	e1_text = StringVar()
	e2_text = StringVar()
	unit_text = StringVar()
	def only_numbers(char):
		return char.isdigit() or char=="."
	def only_char(char):
		return not char.isdigit()
	def char_limit(txt,lim):
		if (len(txt.get())>lim):
			txt.set(txt.get()[:lim])
	vcmd = (measure_window.register(only_numbers),'%S')
	vcmd_char = (measure_window.register(only_char),'%S')
	e1 = Entry(measure_window,width=5,validate="key",validatecommand=vcmd, textvariable = e1_text)
	e2 = Entry(measure_window,width=5,validate="key",validatecommand=vcmd, textvariable = e2_text)
	unit = Entry(measure_window,width=3,validate="key",validatecommand=vcmd_char, textvariable = unit_text)
	e1.grid(row=0, column=1)
	e2.grid(row=1, column=1)
	unit.grid(row=1,column=2)
	if self.distance_unit != "pixels":
		e1_text.set(self.pixel_scaling[0])
		e2_text.set(self.pixel_scaling[1])
		unit_text.set(self.distance_unit)
	e1_text.trace("w", lambda *args: char_limit(e1_text,5))
	e2_text.trace("w", lambda *args: char_limit(e2_text,5))
	unit_text.trace("w", lambda *args: char_limit(unit_text,2))
	Label(measure_window, text="pixels").grid(row=0,column=2)
	def button_react():
		if len(e1_text.get()) < 1:
			messagebox.showerror("Pixel distance missing","You must either measure or enter a value for pixel distance to apply scaling.")
		elif len(unit_text.get()) < 1:
			messagebox.showerror("Unit missing","You must enter a unit for known distance.")
		elif len(e2_text.get()) < 1:
			messagebox.showerror("Known distance missing","You must enter a value for known distance to apply scaling.")
		else:
			self.analysis_data = False
			e2_f = float(e2_text.get())
			e1_f = float(e1_text.get())
			self.pixel_scaling = [e1_f,e2_f,e2_f/e1_f]
			self.distance_unit = unit_text.get().replace('u',u'\u03BC')
			close_measure_window()
			self.gui_update_title()
	def close_measure_window():
		nonlocal prev_rect, prev_draw_img
		if prev_rect:
			self.window_canvas.delete(prev_draw_img)
			self.window_canvas.delete(prev_rect)
			prev_draw_img = False
			prev_rect = False
		measure_window.destroy()
	measure_window.protocol("WM_DELETE_WINDOW", close_measure_window)
	scale_line = False
	scale_start = [0,0]
	prev_img = False
	prev_draw_img = False
	prev_rect = False
	def create_rectangle(x1, y1, x2, y2, alpha=0.5, f ="#7f7f7f"):
		nonlocal prev_img, prev_draw_img, prev_rect
		fill = self.window.winfo_rgb(f) + (int(alpha*255),)
		prev_img = ImageTk.PhotoImage(Image.new('RGBA', (abs(x2-x1), abs(y2-y1)), fill))
		if prev_rect:
			self.window_canvas.delete(prev_draw_img)
			self.window_canvas.delete(prev_rect)
		prev_draw_img = self.window_canvas.create_image(min(x1,x2), y1, image=prev_img, anchor='nw')
		prev_rect = self.window_canvas.create_rectangle(x1, y1, x2, y2)
	def button_measure():
		measure_window.withdraw()
		self.window_canvas.config(cursor="cross")
		self.window_canvas.bind("<Button-1>",gui_scale_line)
	def draw_scale_line(event):
		lim = 1/self.zoom_scale
		start_x = min(max(scale_start[0]-self.zoom_corner[0],0),lim)
		start_y = min(max(scale_start[1]-self.zoom_corner[1],0),lim)
		if start_y-5*lim/self.window_y_size < 0 or start_y+5*lim/self.window_y_size > lim:
			try:
				self.window_canvas.delete(prev_draw_img)
				self.window_canvas.delete(prev_rect)
			except:
				pass
			return False
		start_x = int(start_x*self.zoom_scale*self.window_x_size)
		start_y = int(start_y*self.zoom_scale*self.window_y_size)
		create_rectangle(start_x,start_y-5,event.x,start_y+5)
	def gui_scale_line(event):
		nonlocal scale_line, scale_start
		if not scale_line:
			pos_x = event.x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
			pos_y = event.y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
			scale_line = self.window_canvas.create_line(event.x,event.y,event.x,event.y)
			scale_start = [pos_x,pos_y]
			self.window_canvas.bind('<Motion>', draw_scale_line)
		else:
			self.window_canvas.config(cursor="")
			self.window_canvas.unbind('<Motion>')
			self.window_canvas.unbind('<Button-1>')
			self.window_canvas.delete(scale_line)
			scale_line = False
			pos_x = event.x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
			e1_text.set(int(np.round(abs(scale_start[0]-pos_x)*self.main_image.shape[1])))
			measure_window.deiconify()
			scale_start = [0,0]
	Button(measure_window,text="Apply",command=button_react).grid(row=2,column=0,pady=2)
	Button(measure_window,text="Cancel",command=close_measure_window).grid(row=2,column=1,pady=2)
	Button(measure_window,text="Measure",command=button_measure).grid(row=2,column=2,pady=2,padx=20)
	measure_window.geometry(f"+{(self.screen_width-measure_window.winfo_reqwidth())//2}+{(self.screen_height-measure_window.winfo_reqheight())//2}")
	measure_window.update()
	measure_window.deiconify()

def check_analysis_data(self):
	if type(self.p_centres) == bool and len(self.extra_pts) == 0:
		return False
	if type(self.analysis_data) == bool:
		if len(self.extra_pts) > 0:
			if type(self.original_image) == bool:
				self.analysis_data = denmap_stats.stats(np.append(self.p_centres,self.extra_pts,axis=0),self.pixel_scaling[2])
			else:
				self.analysis_data = denmap_stats.stats(np.append(self.p_centres,self.extra_pts,axis=0),self.pixel_scaling[2],image=self.original_image)
		else:
			if type(self.original_image) == bool:
				self.analysis_data = denmap_stats.stats(self.p_centres,self.pixel_scaling[2])
			else:
				self.analysis_data = denmap_stats.stats(self.p_centres,self.pixel_scaling[2],image=self.original_image)
	return True

def plot_PDAS_hist_axes(self,ax):
	sp = np.concatenate(self.analysis_data.spacings)
	t_10 = np.log10(np.mean(sp))
	t_10 = int(t_10  if t_10  >= 0 else t_10-1)
	low_bound = np.round(np.min(sp),2-t_10)
	low_bound -= low_bound%10**(t_10-1)
	high_bound = np.round(np.max(sp),2-t_10)
	high_bound += 10**(t_10-1)-high_bound%10**(t_10-1)
	freq, bins, _ = ax.hist(sp, np.arange(low_bound,high_bound,10**(t_10-1)), color='#5bcef4', edgecolor='black', linewidth=1)
	bins = 0.5*(np.delete(bins,-1)+np.delete(np.roll(bins,-1),-1))
	ax.set_xlabel("Local Primary Spacing ("+self.distance_unit+")")
	ax.set_ylabel("Local Primary Spacing Frequency")
	[ax.spines[i].set_linewidth(2) for i in ax.spines]
	ax.xaxis.set_tick_params(width=2,direction="in",right=True)
	ax.yaxis.set_tick_params(width=2,direction="in",top=True)
	ax.set_ylim(ax.get_ylim()[0],ax.get_yticks()[-1])
	ax.set_xlim(ax.get_xticks()[np.where(low_bound-ax.get_xticks()>0)[0][-1]],ax.get_xticks()[np.where(high_bound-ax.get_xticks()<0)[0][0]])
	def gaussian_func(x,a,mean,sigma):
		return a*np.exp(-0.5*((x-mean)/sigma)**2)
	coef, _ = curve_fit(gaussian_func,bins,freq,p0=[np.max(freq),np.mean(sp),np.std(sp)])
	new_x = np.linspace(low_bound,high_bound,100)
	ax.plot(new_x,gaussian_func(new_x,*coef),color="black")
	return coef

def gui_PDAS_histogram(self):
	if not self.check_analysis_data():
		return
	pdas_window = Toplevel()
	pdas_window.withdraw()
	pdas_info_window = Toplevel(pdas_window)
	pdas_info_window.withdraw()
	def pdas_close():
		pdas_window.destroy()
		pdas_info_window.destroy()
		plt.close('all')
		if self.highlight_outliers.get() and len(self.highlight_pts) == 0:
			#highlight_outliers is from gui_popup
			self.highlight_outliers.set(0)
	def pdas_info_close():
		pdas_info_window.transient(self.window)
		pdas_info_window.withdraw()
	pdas_window.protocol("WM_DELETE_WINDOW", pdas_close)
	pdas_window.title("Local Primary Spacing Histogram")
	pdas_window.transient(self.window)
	pdas_window.resizable(0,0)
	pdas_window.grab_set()
	pdas_info_window.protocol("WM_DELETE_WINDOW", pdas_info_close)
	pdas_info_window.title("LPS Info")
	pdas_info_window.transient(self.window)
	pdas_info_window.resizable(0,0)
	f = plt.figure(figsize=(5,4))
	canvas = FigureCanvasTkAgg(f, master=pdas_window)
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=3)
	ax = f.add_subplot()
	coef = self.plot_PDAS_hist_axes(ax)
	pdas_window.update()
	pdas_window.geometry(f"+{(self.screen_width-pdas_window.winfo_width())//2}+{(self.screen_height-pdas_window.winfo_height())//2}")
	f.canvas.draw()
	def pdas_right_click(event):
		try:
			self.pdas_popup.tk_popup(event.x_root, event.y_root)
			global pop_x,pop_y
			pop_x = event.x
			pop_y = event.y
		finally:
			popup.grab_release()
	def pdas_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the histogram plot",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			f.canvas.print_png(img_save_name)
	def pdas_information():
		pdas_info_window.transient(pdas_window)
		pdas_info_window.update()
		pdas_info_window.deiconify()
	def pdas_copy():
		pyperclip.copy(f"Mean LPS:\t{self.analysis_data.pdas_mean}\nStD LPS:\t{self.analysis_data.pdas_std}\nGaussian a:\t{coef[0]}\nGaussian mean:\t{coef[1]}\nGaussian sigma:\t{coef[2]}")
	def scale_move_low(pos):
		if float(pos) > float(bar_high.get()):
			bar_low.set(bar_high.get())
			return
		line_id.set_xdata(float(pos))
		canvas.draw()
	def scale_move_high(pos):
		if float(pos) < float(bar_low.get()):
			bar_high.set(bar_low.get())
			return
		line_id2.set_xdata(float(pos))
		canvas.draw()
	def highlight_apply():
		self.highlight_pts = self.analysis_data.spacing_outlier(float(bar_low.get()),bar_high.get())
		if type(self.new_image) != bool:
			if len(self.new_image.shape) < 3:
				self.new_image = cv2.cvtColor(self.new_image,cv2.COLOR_GRAY2RGB)
			self.main_image = self.gui_draw_circles(self.new_image,self.p_centres,return_np=True)
			self.circle_image = Image.fromarray(self.main_image)
		else:
			if len(self.original_image.shape) < 3:
				self.original_image = cv2.cvtColor(self.original_image,cv2.COLOR_GRAY2RGB)
			self.main_image = self.gui_draw_circles(self.original_image,self.p_centres,return_np=True)
			self.circle_image = Image.fromarray(self.main_image)
		width =  1/self.zoom_scale
		x_s = int(np.round(self.zoom_corner[0]*self.main_image.shape[1]))
		x_e = int(np.round((self.zoom_corner[0]+width)*self.main_image.shape[1]))
		y_s = int(np.round(self.zoom_corner[1]*self.main_image.shape[0]))
		y_e = int(np.round((self.zoom_corner[1]+width)*self.main_image.shape[0]))
		cropped_im = self.main_image[y_s:y_e,x_s:x_e]
		cropped_im = cv2.resize(cropped_im,(self.window_x_size,self.window_y_size))
		self.gui_update_image(Image.fromarray(cropped_im))
		if len(self.highlight_pts) > 0:
			self.highlight_outliers.set(1)
		else:
			self.highlight_outliers.set(0)
		pdas_close()
		return
	pdas_window.bind("<Button-3>",pdas_right_click)
	pdas_popup = Menu(pdas_window,tearoff=0)
	pdas_popup.add_command(label="Save plot",command=pdas_save_plot)
	pdas_popup.add_command(label="Information",command=pdas_information)
	low_lim = ax.get_xlim()[0]
	high_lim = ax.get_xlim()[1]
	grad = (high_lim-low_lim)/10000
	Label(pdas_window,text="Highlight outlier areas:").grid(column=1,row=6,sticky=W)
	bar_low = Scale(pdas_window, from_=low_lim, to=high_lim,tickinterval=grad, orient=HORIZONTAL, length=int(f.get_size_inches()[0]*f.dpi), command=scale_move_low)
	bar_low.grid(column = 1,row=7,pady=1,columnspan=3)
	bar_low.set(coef[1]-3*coef[2])
	bar_high = Scale(pdas_window, from_=low_lim, to=high_lim,tickinterval=grad, orient=HORIZONTAL, length=int(f.get_size_inches()[0]*f.dpi), command=scale_move_high)
	bar_high.grid(column = 1,row=8,columnspan=3)
	bar_high.set(coef[1]+3*coef[2])
	ax.axvline(x=coef[1],linestyle='dashed',color='black')
	ax.axvline(x=coef[1]-coef[2],linestyle='dashed',color='black')
	ax.axvline(x=coef[1]-2*coef[2],linestyle='dashed',color='black')
	ax.axvline(x=coef[1]-3*coef[2],linestyle='dashed',color='black')
	ax.axvline(x=coef[1]+coef[2],linestyle='dashed',color='black')
	ax.axvline(x=coef[1]+2*coef[2],linestyle='dashed',color='black')
	ax.axvline(x=coef[1]+3*coef[2],linestyle='dashed',color='black')
	line_id = ax.axvline(x=float(bar_low.get()),color='r')
	line_id2 = ax.axvline(x=float(bar_high.get()),color='r')
	Button(pdas_window,text="Apply",command=highlight_apply).grid(column=2,row=9, sticky=E)
	pdas_window.update()
	pdas_window.deiconify()
	pdas_info_text = f"Mean LPS: {self.analysis_data.pdas_mean:.2f}\nStD LPS: {self.analysis_data.pdas_std:.2f}\nGaussian Fit:\n\ta: {coef[0]:.2f}\n\tmean: {coef[1]:.2f}\n\tsigma: {coef[2]:.2f}"
	info_label = Label(pdas_info_window,text=pdas_info_text,justify=LEFT)
	info_label.pack()
	info_copy = Button(pdas_info_window,text="Copy to clipboard",command=pdas_copy())
	info_copy.pack()
	pdas_info_window.update()
	pdas_info_window.geometry(f"+{(self.screen_width-pdas_info_window.winfo_width())//2}+{(self.screen_height-pdas_info_window.winfo_height())//2}")

def plot_N_hist_axes(self,ax):
	sp = [len(i) for i in self.analysis_data.spacings]
	low_bound = int(np.min(sp))
	low_bound -= low_bound%10
	high_bound = int(np.max(sp))
	high_bound += 10-high_bound%10
	_, bins, _ = ax.hist(sp, np.arange(3,self.analysis_data.max_neigh+1), color='#5bcef4', edgecolor='black', linewidth=1)
	bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
	ax.set_xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w))
	ax.set_xticklabels(bins[:-1])
	#ax.set_xlim(bins[0], bins[-1])
	ax.set_xlabel("Coordination Number (N)")
	ax.set_ylabel("Shape Frequency")
	[ax.spines[i].set_linewidth(2) for i in ax.spines]
	ax.xaxis.set_tick_params(width=2,direction="in",right=True)
	ax.yaxis.set_tick_params(width=2,direction="in",top=True)
	ax.set_ylim(ax.get_ylim()[0],ax.get_yticks()[-1])

def gui_N_histogram(self):
	if not self.check_analysis_data():
		return
	N_window = Toplevel()
	N_window.withdraw()
	def N_close():
		N_window.destroy()
		plt.close('all')
	N_window.protocol("WM_DELETE_WINDOW", N_close)
	N_window.title("Coordination Number Histogram")
	N_window.transient(self.window)
	N_window.grab_set()
	f = plt.figure(figsize=(5,4))
	canvas = FigureCanvasTkAgg(f, master=N_window)
	canvas.get_tk_widget().pack(fill=BOTH, expand=1)
	ax = f.add_subplot()
	self.plot_N_hist_axes(ax)
	N_window.update()
	N_window.geometry("+"+str((self.screen_width-N_window.winfo_width())//2)+"+"+str((self.screen_height-N_window.winfo_height())//2))
	f.canvas.draw()
	def N_resize_plot(event):
		f.set_size_inches(event.width/f.dpi,event.height/f.dpi)
	canvas.mpl_connect("<Configure>",N_resize_plot)
	def N_right_click(event):
		try:
			N_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def N_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the histogram plot",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			f.canvas.print_png(img_save_name)
	N_window.bind("<Button-3>",N_right_click)
	N_popup = Menu(N_window,tearoff=0)
	N_popup.add_command(label="Save plot",command=N_save_plot)
	N_window.update()
	N_window.deiconify()

def plot_K_N_axes(self,ax):
	sp = np.concatenate(self.analysis_data.spacings)
	low_bound = int(np.min(sp))
	low_bound -= low_bound%10
	high_bound = int(np.max(sp))
	high_bound += 10-high_bound%10
	ax.errorbar(np.arange(3,self.analysis_data.max_neigh), self.analysis_data.K_mean,fmt='s', yerr=2*np.array(self.analysis_data.K_std), color='black', linestyle=(0, (5,5)),capsize=4, markersize=4, linewidth=3)
	ax.errorbar(np.arange(3,self.analysis_data.max_neigh), self.analysis_data.K_mean,fmt='s', yerr=2*np.array(self.analysis_data.K_std), color='#5bcef4', linestyle=(-0.5, (9,11)),capsize=0, markersize=3)
	ax.plot(np.arange(3,self.analysis_data.max_neigh),self.analysis_data.K_SLS[3:self.analysis_data.max_neigh],color="black")
	ax.set_xlabel("Coordination Number (N)")
	ax.set_ylabel("Packing Factor (K)")
	[ax.spines[i].set_linewidth(2) for i in ax.spines]
	ax.xaxis.set_tick_params(width=2,direction="in",right=True)
	ax.yaxis.set_tick_params(width=2,direction="in",top=True)
	ax.set_ylim(ax.get_ylim()[0],ax.get_yticks()[-1])

def gui_K_N(self):
	if not self.check_analysis_data():
		return
	K_window = Toplevel()
	K_window.withdraw()
	def K_close():
		K_window.destroy()
		plt.close('all')
	K_window.protocol("WM_DELETE_WINDOW", K_close)
	K_window.title("K Histogram")
	K_window.transient(self.window)
	K_window.grab_set()
	f = plt.figure(figsize=(5,4))
	canvas = FigureCanvasTkAgg(f, master=K_window)
	canvas.get_tk_widget().pack(fill=BOTH, expand=1)
	ax = f.add_subplot()
	self.plot_K_N_axes(ax)
	K_window.update()
	K_window.geometry(f"+{(self.screen_width-K_window.winfo_width())//2}+{(self.screen_height-K_window.winfo_height())//2}")
	f.canvas.draw()
	def K_resize_plot(event):
		f.set_size_inches(event.width/f.dpi,event.height/f.dpi)
	canvas.mpl_connect("<Configure>",K_resize_plot)
	def K_right_click(event):
		try:
			K_popup.tk_popup(event.x_root, event.y_root)
			global pop_x,pop_y
			pop_x = event.x
			pop_y = event.y
		finally:
			popup.grab_release()
	def K_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the K vs N plot",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			f.canvas.print_png(img_save_name)
	K_window.bind("<Button-3>",K_right_click)
	K_popup = Menu(K_window,tearoff=0)
	K_popup.add_command(label="Save plot",command=K_save_plot)
	K_window.update()
	K_window.deiconify()

def plot_PDAS_N_axes(self, ax):
	sp_N = np.array([len(i) for i in self.analysis_data.spacings])
	sp_N_num = np.array([i*np.sum(sp_N == i) for i in range(3,self.analysis_data.max_neigh)])
	sp_mean = [np.mean(np.concatenate(self.analysis_data.spacings[sp_N == i])) for i in range(3,self.analysis_data.max_neigh) if np.sum(sp_N == i) > 0]
	sp_min = [np.sort(np.concatenate(self.analysis_data.spacings[sp_N == i]))[int(sp_N_num[i-3]*0.1)] for i in range(3,self.analysis_data.max_neigh) if np.sum(sp_N == i) > 0]
	sp_max = [np.sort(np.concatenate(self.analysis_data.spacings[sp_N == i]))[int(sp_N_num[i-3]*0.9)] for i in range(3,self.analysis_data.max_neigh) if np.sum(sp_N == i) > 0]
	keep_N = [(np.sum(sp_N == i) > 0) for i in range(3,self.analysis_data.max_neigh)]
	N_plot = np.arange(3,self.analysis_data.max_neigh)[keep_N]
	ax.scatter(N_plot,sp_mean,color="black",marker="*")
	ax.scatter(N_plot,sp_max,color="red",marker="*")
	ax.scatter(N_plot,sp_min,color="blue",marker="*")
	ax.set_ylabel("Local Primary Spacing ("+self.distance_unit+")")
	ax.set_xlabel("Coordination Number (N)")
	[ax.spines[i].set_linewidth(2) for i in ax.spines]
	ax.xaxis.set_tick_params(width=2,direction="in",right=True)
	ax.yaxis.set_tick_params(width=2,direction="in",top=True)
	ax.set_ylim(ax.get_yticks()[0],ax.get_yticks()[-1])
	fit_min = linregress(N_plot,sp_min)
	fit_max = linregress(N_plot,sp_max)
	fit_mean = linregress(N_plot,sp_mean)
	ax.plot(N_plot,N_plot*fit_mean.slope+fit_mean.intercept,linestyle="dashed",color="black")
	ax.plot(N_plot,N_plot*fit_min.slope+fit_min.intercept,linestyle="dashed",color="blue")
	ax.plot(N_plot,N_plot*fit_max.slope+fit_max.intercept,linestyle="dashed",color="red")
	return fit_mean, fit_min, fit_max

def gui_PDAS_N(self):
	if not self.check_analysis_data():
		return
	pdas_n_window = Toplevel()
	pdas_n_window.withdraw()
	pdas_n_info_window = Toplevel(pdas_n_window)
	pdas_n_info_window.withdraw()
	def pdas_n_close():
		pdas_n_window.destroy()
		pdas_n_info_window.destroy()
		plt.close('all')
	def pdas_n_info_close():
		pdas_n_info_window.transient(self.window)
		pdas_n_info_window.withdraw()
	pdas_n_window.protocol("WM_DELETE_WINDOW", pdas_n_close)
	pdas_n_window.title("Local Primary Spacing vs N plot")
	pdas_n_window.transient(self.window)
	pdas_n_window.grab_set()
	pdas_n_info_window.protocol("WM_DELETE_WINDOW", pdas_n_info_close)
	pdas_n_info_window.title("Local Primary Spacing vs N Info")
	pdas_n_info_window.transient(self.window)
	pdas_n_info_window.resizable(0,0)
	f = plt.figure(figsize=(5,4))
	canvas = FigureCanvasTkAgg(f, master=pdas_n_window)
	canvas.get_tk_widget().pack(fill=BOTH, expand=1)
	ax = f.add_subplot()
	fit_mean, fit_min, fit_max = self.plot_PDAS_N_axes(ax)
	pdas_n_window.update()
	pdas_n_window.geometry(f"+{(self.screen_width-pdas_n_window.winfo_width())//2}+{(self.screen_height-pdas_n_window.winfo_height())//2}")
	f.canvas.draw()
	def pdas_n_resize_plot(event):
		f.set_size_inches(event.width/f.dpi,event.height/f.dpi)
	canvas.mpl_connect("<Configure>",pdas_n_resize_plot)
	def pdas_n_right_click(event):
		try:
			pdas_n_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def pdas_n_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the histogram plot",filetypes=[("Portable Network Graphics",".png")],defaultextension=".png")
		if img_save_name:
			f.canvas.print_png(img_save_name)
	def pdas_n_information():
		pdas_n_info_window.transient(pdas_n_window)
		pdas_n_info_window.update()
		pdas_n_info_window.deiconify()
	def pdas_n_copy():
		pyperclip.copy(
			"Max m:\t"+str(fit_max.slope)+"\nMax c:\t"+str(fit_max.intercept)+"\nMax R value:\t"+str(fit_max.rvalue)+"\nMax std error:\t"+str(fit_max.stderr)+
			"\nMean m:\t"+str(fit_mean.slope)+"\nMean c:\t"+str(fit_mean.intercept)+"\nMean R value:\t"+str(fit_mean.rvalue)+"\nMean std error:\t"+str(fit_mean.stderr)+
			"\nMin m:\t"+str(fit_min.slope)+"\nMin c:\t"+str(fit_min.intercept)+"\nMin R value:\t"+str(fit_min.rvalue)+"\nMin std error:\t"+str(fit_min.stderr)
		)
	pdas_n_window.bind("<Button-3>",pdas_n_right_click)
	pdas_n_popup = Menu(pdas_n_window,tearoff=0)
	pdas_n_popup.add_command(label="Save plot",command=pdas_n_save_plot)
	pdas_n_popup.add_command(label="Information",command=pdas_n_information)
	pdas_n_window.update()
	pdas_n_window.deiconify()
	pdas_n_info_text = (
		"Max Slope: {:.2f}".format(fit_max.slope)+"\nMax Intercept: {:.2f}".format(fit_max.intercept)+"\nMax R value: {:.2f}".format(fit_max.rvalue)+"\nMax std error: {:.2f}".format(fit_max.stderr)+
		"\n\nMean Slope: {:.2f}".format(fit_mean.slope)+"\nMean Intercept: {:.2f}".format(fit_mean.intercept)+"\nMean R value: {:.2f}".format(fit_mean.rvalue)+"\nMean std error: {:.2f}".format(fit_mean.stderr)+
		"\n\nMin Slope: {:.2f}".format(fit_min.slope)+"\nMin Intercept: {:.2f}".format(fit_min.intercept)+"\nMin R value: {:.2f}".format(fit_max.rvalue)+"\nMin std error: {:.2f}".format(fit_min.stderr)
	)
	info_label = Label(pdas_n_info_window,text=pdas_n_info_text,justify=LEFT)
	info_label.pack()
	info_copy = Button(pdas_n_info_window,text="Copy to clipboard",command=pdas_n_copy())
	info_copy.pack()
	pdas_n_info_window.update()
	pdas_n_info_window.geometry(f"+{(self.screen_width-pdas_n_info_window.winfo_width())//2}+{(self.screen_height-pdas_n_info_window.winfo_height())//2}")