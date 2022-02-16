from tkinter import Toplevel, Label, StringVar, Entry, Button, Scale, messagebox, RIGHT, HORIZONTAL, END
from tkinter.ttk import Progressbar
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .gui_mp_functions import progress_fft, auto_process_thread
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import denmap_src.width_gui as wg
import cv2


def gui_auto_process(self):
    if not type(self.main_image) == bool and not self.thr.is_alive():
        self.create_progressbar(500)
        self.progress_bar.title("Starting NCC processing")
        self.window.update()
        self.window.after(50,self.gui_update_progress)
        self.thr = mp.Process( target=auto_process_thread, args = (self.new_image, self.original_image, self.circle_image, self.binary_image, self.res_queue, self.prog_queue, self.shared_memory))
        self.thr.start()
    else:
        messagebox.showerror("No loaded image","You must load an image before it can be processed!")

def gui_update_progress(self):
	while not self.prog_queue.empty():
		i = self.prog_queue.get()
		self.progress['value']=i[0]
		self.progress_bar.title(i[1])
	if (self.res_queue.empty()):
		self.window.after(50,self.gui_update_progress)
	else:
		data = self.res_queue.get()
		if type(data[0]) == int:
			if data[0] == -1:
				messagebox.showerror("Unable to find templates","The program was unable to find any dendritic templates!")
				self.progress_bar.destroy()
				return
			elif data[0] == -2:
				messagebox.showerror("Bandpass Filter Failed!","Your PC does not have enough Memory!")
				self.progress_bar.destroy()
				return
			else:
				self.progress_bar.destroy()
				return
		self.new_image = data[0].astype(np.uint8)
		self.NCC_values = data[1]
		self.NCC_thres = data[2]
		self.NCC_temp = data[3]
		self.p_centres = data[4]
		self.circle_image = self.gui_draw_circles(Image.fromarray(self.new_image),self.p_centres)
		self.main_image = np.array(self.circle_image)
		self.binary_image = wg.width_binary(self.new_image,wg.detect_image_thres(self.new_image))
		self.highlight_pts = []
		self.gui_finish_process()
		self.gui_enable_menu()

def gui_finish_process(self):
	self.progress_bar.destroy()
	if not type(self.circle_image) == bool:
		self.gui_update_image(self.circle_image)
		self.zoom_scale = 1
		self.zoom_corner = [0,0]
		last = self.menu_view.index(END)+1
		for i in range(0,last):
			self.menu_view.entryconfig(i, state="normal")
			self.image_checkvariables[i].set(0)
		self.image_checkvariables[3].set(1)

def create_progressbar(self,leng):
    self.progress_bar = Toplevel(self.window)
    self.progress = Progressbar(self.progress_bar, orient = HORIZONTAL, length = leng, mode = 'indeterminate')
    self.progress_bar.geometry(f"+{(self.screen_width-self.progress['length'])//2}+{(self.screen_height-self.progress_bar.winfo_reqheight())//2}")
    self.progress.pack()
    self.progress_bar.transient(self.window)
    self.progress_bar.grab_set()
    self.progress_bar.protocol("WM_DELETE_WINDOW", self.do_nothing)
    self.progress_bar.resizable(0,0)
    #progress_bar.attributes('-disabled', True)


def gui_fft_thres(self):
    if not type(self.original_image) == bool:
        if type(self.FFT_in) == bool:
            if len(self.original_image.shape) < 3:
                self.FFT_in = wg.calculate_r_in(self.original_image)
            else:
                self.original_image = self.original_image[:,:,0]
                self.FFT_in = wg.calculate_r_in(self.original_image)
        thres_win = Toplevel()
        thres_win.title("FFT Bandpass")
        thres_win.resizable(0,0)
        thres_win.transient(self.window)
        thres_win.grab_set()
        Label(thres_win, text="FFT filter pixels up to:",justify=RIGHT).grid(row=0)
        Label(thres_win, text="FFT filter pixels down to:",justify=RIGHT).grid(row=1)
        e1_text = StringVar()
        e2_text = StringVar()
        def only_numbers(char):
            return char.isdigit()
        def char_limit(txt):
            if (len(txt.get())>3):
                txt.set(txt.get()[:3])
        vcmd = (thres_win.register(only_numbers),'%S')
        e1 = Entry(thres_win,width=3,validate="key",validatecommand=vcmd, textvariable = e1_text)
        e2 = Entry(thres_win,width=3,validate="key",validatecommand=vcmd, textvariable = e2_text)
        e1.grid(row=0, column=1)
        e2.grid(row=1, column=1)
        e1_text.set(str(self.FFT_in))
        e2_text.set(str(self.FFT_out))
        e1_text.trace("w", lambda *args: char_limit(e1_text))
        e2_text.trace("w", lambda *args: char_limit(e2_text))
        Label(thres_win, text="pixels").grid(row=0,column=2)
        Label(thres_win, text="pixels").grid(row=1,column=2)
        def button_react():
            try:
                self.FFT_in = int(e1_text.get())
                self.FFT_out = int(e2_text.get())
                mp.Process( target=progress_fft, args = (self.original_image, self.FFT_in, self.FFT_out, self.prog_queue, self.res_queue, self.shared_memory)).start()
                self.create_progressbar(500)
                self.progress_bar.title("Performing FFT Bandpass filter")
                self.window.after(50,self.gui_update_fft_progress)
                thres_win.destroy()
            except:
            	self.FFT_in = min(int(np.round(2.5e-3*np.sqrt(self.original_image.shape[0]*self.original_image.shape[1]))),50)
            	self.FFT_out = 100
            	messagebox.showerror("Invalid pixel size","Could not convert the given pixels into integers!")
            
        Button(thres_win,text="Apply",command=button_react).grid(row=2,column=0,pady=2)
        Button(thres_win,text="Cancel",command=lambda: thres_win.destroy()).grid(row=2,column=1,pady=2)
        thres_win.geometry(f"+{(self.screen_width-thres_win.winfo_reqwidth())//2}+{(self.screen_height-thres_win.winfo_reqheight())//2}")
    else:
        messagebox.showerror("Image not loaded","You must load an image before performing a bandpass filter.")


def gui_update_fft_progress(self):
	while not self.prog_queue.empty():
		i = self.prog_queue.get()
		self.progress['value']=i[0]
		self.progress_bar.title(i[1])
	if (self.res_queue.empty()):
		self.window.after(50,self.gui_update_fft_progress)
	else:
		self.new_image = self.res_queue.get()
		if (type(self.new_image) == int):
			if (self.new_image == -1):
				messagebox.showerror("Bandpass Filter Failed!","Your PC does not have enough Memory!")
				self.progress_bar.destroy()
				return
		self.gui_update_image(Image.fromarray(self.new_image))
		self.main_image = self.new_image
		self.binary_image = False
		self.circle_image = False
		self.progress_bar.destroy()
		self.zoom_scale = 1
		self.zoom_corner = [0,0]
		self.unset_view_variables()
		self.menu_view.entryconfig(1, state="disabled")
		self.menu_view.entryconfig(3, state="disabled")
		self.menu_view.entryconfig(2, state="normal")
		self.image_checkvariables[2].set(1)

def gui_binary(self):
	if not type(self.main_image) == bool:
		image = self.main_image
	else:
		messagebox.showerror("Image not loaded","You must load an image before attempting to convert image to binary!")
		return
	self.zoom_scale = 1
	self.zoom_corner = [0,0]
	self.gui_update_image(Image.fromarray(self.main_image))
	binary_window = Toplevel()
	def binary_close():
		binary_window.destroy()
		plt.close('all')
		self.gui_update_image(Image.fromarray(image))
	binary_window.protocol("WM_DELETE_WINDOW", binary_close)
	binary_window.title("Create binary image")
	binary_window.transient(self.window)
	binary_window.grab_set()
	f = plt.figure(figsize=(5,4))
	canvas = FigureCanvasTkAgg(f, master=binary_window)
	canvas.get_tk_widget().grid(row=0, rowspan=6, columnspan=3)
	ax = f.add_axes((0, 0, 1, 1))
	ax.axis("off")
	ax.margins(0,tight=True)
	hist = cv2.calcHist([image],[0],None,[256],[0,256]).T[0].astype(int)
	hist[-2] += hist[-1]
	hist = hist[:-1]
	bins = np.arange(256)
	_, _, _ = ax.hist(bins[:-1], bins,weights=hist, color='black')
	n_hist = hist.copy()
	n_hist[:2] = 0
	n_hist[-2:] = 0
	n_hist = np.delete(n_hist,np.arange(110,128))
	ax.set_ylim(top=1.1*max(n_hist))
	
	def scale_move(val):
		line_id.set_xdata(int(val))
		canvas.draw()

	def preview_button():
		binary = wg.width_binary(image,int(bar.get()))
		self.gui_update_image(Image.fromarray(binary))

	def apply_button():
		self.binary_image = wg.width_binary(image,int(bar.get()))
		self.gui_update_image(Image.fromarray(self.binary_image))
		self.main_image = self.binary_image
		binary_window.destroy()
		plt.close('all')
		self.unset_view_variables()
		self.menu_view.entryconfig(1, state="normal")
		self.image_checkvariables[1].set(1)

	bar = Scale(binary_window, from_=0, to=255, orient=HORIZONTAL, length=int(f.get_size_inches()[0]*f.dpi), command=scale_move)
	bar.grid(row=7,pady=1,columnspan=3)
	bar.set(wg.detect_image_thres(image))
	line_id = ax.axvline(x=int(bar.get()),color='r')
	Button(binary_window,text="Cancel",command=binary_close).grid(row=8,column=2,pady=2)
	Button(binary_window,text="Preview",command=preview_button).grid(row=8,column=1,pady=2)
	Button(binary_window,text="Apply",command=apply_button).grid(row=8,column=0,pady=2)
	binary_window.resizable(False, False)
	f.set_size_inches

def gui_NCC(self):
	if type(self.NCC_values) == bool:
		messagebox.showerror("Image not processed","You must process the image to find the NCC values!")
		return
	self.zoom_scale = 1
	self.zoom_corner = [0,0]
	self.gui_update_image(Image.fromarray(self.main_image))
	NCC_window = Toplevel()
	def NCC_close():
		NCC_window.destroy()
		plt.close('all')
		self.gui_update_image(self.circle_image)
	global circle_image, p_centres
	NCC_window.protocol("WM_DELETE_WINDOW", NCC_close)
	NCC_window.title("Define NCC Threshold")
	NCC_window.transient(self.window)
	NCC_window.grab_set()
	f = plt.figure(figsize=(5,4))
	canvas = FigureCanvasTkAgg(f, master=NCC_window)
	canvas.get_tk_widget().grid(row=0, rowspan=6, columnspan=3)
	ax = f.add_axes((0, 0, 1, 1))
	ax.axis("off")
	ax.margins(0,tight=True)
	hist, bins = np.histogram(self.NCC_values.reshape(1,-1),1000)
	bins = (np.delete(bins,len(bins)-1)+np.delete(bins,0))/2
	ax.plot(bins[2:], hist[2:], color='black')
	def scale_move(val):
		line_id.set_xdata(float(val))
		canvas.draw()
	bar = Scale(NCC_window, from_=0, to=1, orient=HORIZONTAL, length=int(f.get_size_inches()[0]*f.dpi), command=scale_move, resolution=0.001)
	bar.grid(row=7,pady=1,columnspan=3)
	bar.set(self.NCC_thres)
	line_id = ax.axvline(x=float(bar.get()),color='r')
	def preview_button():
		centres = wg.width_find_centres(self.new_image,self.NCC_values,float(bar.get()))
		centres = wg.width_contrast(self.new_image,self.NCC_temp,centres)
		centres = wg.width_distance(self.NCC_values,centres)
		self.gui_update_image(self.gui_draw_circles(Image.fromarray(self.new_image),centres))

	def apply_button():
		self.p_centres = wg.width_find_centres(self.new_image,self.NCC_values,float(bar.get()))
		self.p_centres = wg.width_contrast(self.new_image,self.NCC_temp,self.p_centres)
		self.p_centres = wg.width_distance(self.NCC_values,self.p_centres)
		self.circle_image = self.gui_draw_circles(Image.fromarray(self.new_image),self.p_centres)
		self.main_image = np.array(self.circle_image)
		self.gui_update_image(self.circle_image)
		NCC_close()

	Button(NCC_window,text="Cancel",command=NCC_close).grid(row=8,column=2,pady=2)
	Button(NCC_window,text="Preview",command=preview_button).grid(row=8,column=1,pady=2)
	Button(NCC_window,text="Apply",command=apply_button).grid(row=8,column=0,pady=2)