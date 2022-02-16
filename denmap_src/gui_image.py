import cv2
import numpy as np
from PIL import ImageTk, Image
from tkinter import Toplevel, Button, Label, Entry, StringVar, NW, END, RIGHT, W

def gui_draw_circles(self,image,centres,rad=25,autorad=True,return_np=False):
	img = image.copy()
	if type(img) == Image.Image:
		img = np.array(img)
	if len(img.shape) != 3:
		img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	if (autorad):
		try:
			rad = np.round((self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/8).astype(int)
		except:
			pass
	if type(centres) == bool:
		centres = []
	for p in centres:
		cv2.circle(img,tuple(p),rad,(255,0,0),-1)
	for p in np.array(self.extra_pts):
		cv2.circle(img,tuple(p),rad,(0,255,0),-1)
	for p in np.array(self.highlight_pts):
		cv2.circle(img,tuple(p),rad,(0,0,255),-1)
	if return_np:
		return img
	return Image.fromarray(img)

def gui_update_image(self,image):
	scale = min(0.9*self.screen_width/image.size[0],0.9*self.screen_height/image.size[1])
	im_w = int(image.size[0]*scale)
	im_h = int(image.size[1]*scale)
	self.window_x_size = im_w
	self.window_y_size = im_h
	self.window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
	self.display_image = ImageTk.PhotoImage(image.resize((im_w,im_h)))
	self.window_canvas.delete("all")
	self.canvas_image = self.window_canvas.create_image(0,0,anchor=NW,image=self.display_image)

def gui_scale(self):
	if not type(self.original_image) == bool:
		scale_win = Toplevel()
		scale_win.title("Original Image Scale Resolution")
		scale_win.resizable(0,0)
		scale_win.transient(self.window)
		scale_win.grab_set()
		Label(scale_win, text="Multiply image resolution by:").grid(row=0,columnspan=1)
		e1_text = StringVar()
		def only_numbers(char):
			return char.isdigit() or char=="."
		def char_limit(txt):
			if (len(txt.get())>3):
				txt.set(txt.get()[:3])
			else:
				lab_txt.set(f"Expected Resolution: {int(self.original_image.shape[1]*float(txt.get()))}x{int(self.original_image.shape[0]*float(txt.get()))}")
		vcmd = (scale_win.register(only_numbers),'%S')
		e1 = Entry(scale_win,width=3,validate="key",validatecommand=vcmd, textvariable = e1_text)
		e1.grid(row=0, column=1)
		e1_text.set("1")
		e1_text.trace("w", lambda *args: char_limit(e1_text))
		def button_react():
			y_s = int(self.original_image.shape[0]*float(e1.get()))
			x_s = int(self.original_image.shape[1]*float(e1.get()))
			self.main_image = cv2.resize(self.original_image,(x_s,y_s))
			self.original_image = self.main_image
			self.gui_update_image(Image.fromarray(self.main_image))
			self.gui_update_title()
			scale_win.destroy()
			self.new_image = False
			self.binary_image = False
			self.circle_image = False
			last = self.menu_view.index(END)+1
			for i in range(1,last):
				self.menu_view.entryconfig(i, state="disabled")
				self.image_checkvariables[i].set(0)
			self.menu_view.entryconfig(0, state="normal")
			self.image_checkvariables[0].set(1)
			self.zoom_scale = 1
			self.zoom_corner = [0,0]
		lab_txt = StringVar()
		lab_txt.set(f"Expected Resolution: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
		Label(scale_win, textvariable=lab_txt).grid(row=1,columnspan=2)
		Button(scale_win,text="Apply",command=button_react,justify=RIGHT).grid(row=2,column=0,pady=2,sticky=W,padx=2)
		Button(scale_win,text="Cancel",command=lambda: scale_win.destroy()).grid(row=2,column=1,pady=2,padx=2)
		scale_win.geometry(f"+{(self.screen_width-scale_win.winfo_reqwidth())//2}+{(self.screen_height-scale_win.winfo_reqheight())//2}")
	else:
		messagebox.showerror("Image not loaded","You must load an image before performing a bandpass filter.")
