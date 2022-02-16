from tkinter import Menu, BooleanVar, DISABLED
from PIL import Image
import denmap_src.width_gui as wg
import numpy as np
import cv2

def create_popup_menu(self):
	self.popup = Menu(self.window,tearoff=0)
	self.popup.add_command(label="Add Point",command=self.popup_add_point)
	self.popup.add_command(label="Remove Point",command=self.popup_remove_point)
	self.cont_add = BooleanVar()
	self.popup.add_checkbutton(label="Continuous Add Points",command=self.popup_cont_add_points,offvalue=0, onvalue=1, variable=self.cont_add)
	self.highlight_outliers = BooleanVar()
	self.popup.add_checkbutton(label="Highlight Outliers",command=self.popup_highlight,offvalue=0, onvalue=1, variable=self.highlight_outliers)
	#popup.add_command(label="Continuous remove points",command=popup_cont_remove_points)

def gui_right_click(self,event):
    try:
        self.popup.tk_popup(event.x_root, event.y_root)
        self.pop_x = event.x
        self.pop_y = event.y
    finally:
        self.popup.grab_release()

def popup_add_point(self):
	add_p = True
	global circle_image, main_image, extra_pts, FFT_in, new_image, original_image
	if not type(self.p_centres) == bool:
		if not self.image_checkvariables[3].get():
			self.image_checkvariables[3].set(1)
			self.gui_change_filt_cent(False)
		if type(self.NCC_temp) != bool:
			rad = (self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/10
		else:
			if type(self.FFT_in) == bool:
				if len(self.original_image.shape) < 3:
					self.FFT_in = wg.calculate_r_in(self.original_image)
				else:
					self.FFT_in = wg.calculate_r_in(self.original_image[:,:,0])
				rad = self.FFT_in*1.3
			else:
				rad = self.FFT_in*1.3
		pos_x = self.pop_x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
		pos_y = self.pop_y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
		pos_x = pos_x*self.main_image.shape[1]
		pos_y = pos_y*self.main_image.shape[0]
		r = np.array(self.p_centres)-np.array([pos_x,pos_y])
		r = np.sqrt(np.sum(r**2,axis=1))
		wh_r = np.where(r <= rad)[0]
		if len(wh_r) > 0:
			add_p = False
		if len(self.extra_pts) > 0:
			r = np.array(self.extra_pts)-np.array([pos_x,pos_y])
			r = np.sqrt(np.sum(r**2,axis=1))
			wh_r = np.where(r <= rad)[0]
			if len(wh_r) > 0:
				add_p = False
	elif len(self.extra_pts) > 0:
		wh_r = []
		if type(self.NCC_temp) != bool:
			rad = (self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/10
		else:
			if type(self.FFT_in) == bool:
				if len(self.original_image.shape) < 3:
					self.FFT_in = wg.calculate_r_in(original_image)
				else:
					self.FFT_in = wg.calculate_r_in(self.original_image[:,:,0])
				rad = self.FFT_in*1.3
			else:
				rad = self.FFT_in*1.3
		pos_x = self.pop_x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
		pos_y = self.pop_y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
		pos_x = pos_x*self.main_image.shape[1]
		pos_y = pos_y*self.main_image.shape[0]
		r = np.array(self.extra_pts)-np.array([pos_x,pos_y])
		r = np.sqrt(np.sum(r**2,axis=1))
		wh_r_extra = np.where(r <= rad)[0]
		if len(wh_r_extra) > 0:
			add_p = False
	if add_p:
		pos_x = self.pop_x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
		pos_y = self.pop_y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
		pos_x = pos_x*self.main_image.shape[1]
		pos_y = pos_y*self.main_image.shape[0]
		if len(self.extra_pts) == 0:
			self.extra_pts = [[int(pos_x),int(pos_y)]]
		else:
			self.extra_pts = np.append(self.extra_pts,[[int(pos_x),int(pos_y)]],axis=0)
		#extra_pts += [[int(pos_x),int(pos_y)]]
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
		self.analysis_data = False
		if self.menu_export.entryconfig(3)['state'][-1] == "disabled":
			self.menu_export.entryconfig(3,state="normal")
			self.menu_view.entryconfig(3,state="normal",label='Image with Centres')
			self.menu_process.entryconfig(0,state=DISABLED)
			self.menu_process.entryconfig(1,state=DISABLED)
			self.menu_process.entryconfig(2,state=DISABLED)
			self.image_checkvariables[0].set(0)
			self.image_checkvariables[3].set(1)
		elif self.image_checkvariables[3].get() == 0:
			self.image_checkvariables[0].set(0)
			self.image_checkvariables[1].set(0)
			self.image_checkvariables[2].set(0)
			self.image_checkvariables[3].set(1)

def popup_remove_point(self):
	rem_p = False
	if not type(self.p_centres) == bool:
		if not self.image_checkvariables[3].get():
			self.image_checkvariables[3].set(1)
			self.gui_change_filt_cent(False)
		if type(self.NCC_temp) != bool:
			rad = (self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/10
		else:
			if type(self.FFT_in) == bool:
				if len(self.original_image.shape) < 3:
					self.FFT_in = wg.calculate_r_in(self.original_image)
				else:
					self.FFT_in = wg.calculate_r_in(self.original_image[:,:,0])
				rad = self.FFT_in*1.3
			else:
				rad = self.FFT_in*1.3
		pos_x = self.pop_x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
		pos_y = self.pop_y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
		pos_x = pos_x*self.main_image.shape[1]
		pos_y = pos_y*self.main_image.shape[0]
		r = np.array(self.p_centres)-np.array([pos_x,pos_y])
		r = np.sqrt(np.sum(r**2,axis=1))
		wh_r = np.where(r <= rad)[0]
		if len(wh_r) > 0:
			rem_p = True
		elif len(self.extra_pts) > 0:
			r = np.array(self.extra_pts)-np.array([pos_x,pos_y])
			r = np.sqrt(np.sum(r**2,axis=1))
			wh_r_extra = np.where(r <= rad)[0]
			if len(wh_r_extra) > 0:
				rem_p = True
	elif len(self.extra_pts) > 0:
		wh_r = []
		if type(self.NCC_temp) != bool:
			rad = (self.NCC_temp.shape[0]+self.NCC_temp.shape[1])/10
		else:
			if type(self.FFT_in) == bool:
				if len(self.original_image.shape) < 3:
					self.FFT_in = wg.calculate_r_in(self.original_image)
				else:
					self.FFT_in = wg.calculate_r_in(self.original_image[:,:,0])
				rad = self.FFT_in*1.3
			else:
				rad = self.FFT_in*1.3
		pos_x = self.pop_x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
		pos_y = self.pop_y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
		pos_x = pos_x*self.main_image.shape[1]
		pos_y = pos_y*self.main_image.shape[0]
		r = np.array(self.extra_pts)-np.array([pos_x,pos_y])
		r = np.sqrt(np.sum(r**2,axis=1))
		wh_r_extra = np.where(r <= rad)[0]
		if len(wh_r_extra) > 0:
			rem_p = True
	if rem_p:
		if len(wh_r) > 0:
			self.p_centres = np.delete(self.p_centres,wh_r[0],axis=0)
		elif len(wh_r_extra) > 0:
			self.extra_pts = np.delete(self.extra_pts,wh_r_extra[0],axis=0)
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
		self.analysis_data = False

def unbind_left_click(self, event):
	self.window.unbind("<Button-1>")

def cont_add_point(self, event):
	self.pop_x = event.x
	self.pop_y = event.y
	self.popup_add_point()

def cont_add_control_release(self, event):
	self.window.bind("<Button-1>",self.cont_add_point)

def bind_cont_add(self):
	self.window.bind("<KeyPress-Control_L>",self.unbind_left_click)
	self.window.bind("<KeyRelease-Control_L>",self.cont_add_control_release)
	self.window.bind("<Button-1>",self.cont_add_point)

def unbind_cont_add(self):
	self.window.unbind("<KeyPress-Control_L>")
	self.window.unbind("<KeyRelease-Control_L>")
	self.unbind_left_click(None)

def popup_cont_add_points(self):
	if self.cont_add.get():
		self.bind_cont_add()
	else:
		self.unbind_cont_add()

def popup_highlight(self):
	if self.highlight_outliers.get():
		self.gui_PDAS_histogram()
	else:
		if len(self.highlight_pts) > 0:
			self.highlight_pts = []
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