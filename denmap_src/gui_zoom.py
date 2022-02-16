from PIL import Image
import numpy as np
import cv2

def zoom_or_update(self):
	if self.zoom_scale > 1:
		self.gui_zoom(self.window.winfo_reqwidth()//2,self.window.winfo_reqheight()//2,activate_zoom=False)
	else:
		self.gui_update_image(Image.fromarray(self.main_image))

def gui_zoom(self,x,y,zoom=True,activate_zoom=True):
	if not type(self.main_image) == bool:
		pos_x = x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
		pos_y = y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
		rel_x = x/self.window_x_size
		rel_y = y/self.window_y_size
		if activate_zoom:
			if (zoom):
				self.zoom_scale += 1
			elif (self.zoom_scale > 1):
				self.zoom_scale -= 1
			else:
				return
			width =  1/self.zoom_scale
			self.zoom_corner = [pos_x-rel_x*width,pos_y-rel_y*width]
			if (self.zoom_corner[0]<0):
				self.zoom_corner[0] = 0
			elif (self.zoom_corner[0]+width>1):
				self.zoom_corner[0] = 1-width
				if (self.zoom_corner[0]<0):
					self.zoom_corner[0] = 0
			
			if (self.zoom_corner[1]<0):
				self.zoom_corner[1] = 0
			elif (self.zoom_corner[1]+width>1):
				self.zoom_corner[1] = 1-width
				if (self.zoom_corner[1]<0):
					self.zoom_corner[1] = 0
		else:
			width =  1/self.zoom_scale
		x_s = int(np.round(self.zoom_corner[0]*self.main_image.shape[1]))
		x_e = int(np.round((self.zoom_corner[0]+width)*self.main_image.shape[1]))
		y_s = int(np.round(self.zoom_corner[1]*self.main_image.shape[0]))
		y_e = int(np.round((self.zoom_corner[1]+width)*self.main_image.shape[0]))
		if (x_e-x_s < 5):
			return
		elif (y_e-y_s < 5):
			return
		cropped_im = self.main_image[y_s:y_e,x_s:x_e]
		cropped_im = cv2.resize(cropped_im,(self.window_x_size,self.window_y_size))
		self.gui_update_image(Image.fromarray(cropped_im))

def gui_mouse_wheel(self,event):
    if event.num == 5 or event.delta == -120:
        self.gui_zoom(event.x,event.y,False)
    elif event.num == 4 or event.delta == 120:
        self.gui_zoom(event.x,event.y)

def gui_control_corner(self, event):
	pos_x = event.x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0]
	pos_y = event.y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]
	self.zoom_corner = [self.zoom_corner[0]-pos_x+self.shift_motion_position[0],self.zoom_corner[1]-pos_y+self.shift_motion_position[1]]
	width = 1/self.zoom_scale
	zoom_corner = [min(max(self.zoom_corner[0],0),1-width),min(max(self.zoom_corner[1],0),1-width)]
	self.gui_zoom(event.x,event.y,activate_zoom=False)

def gui_control_press(self, event):
	if self.zoom_scale > 1:
		self.shift_motion_position = [event.x/(self.zoom_scale*self.window_x_size) + self.zoom_corner[0],event.y/(self.zoom_scale*self.window_y_size) + self.zoom_corner[1]]
		self.shift_motion_event = self.window_canvas.bind('<Motion>', self.gui_control_corner)
		self.window_canvas.config(cursor="hand2")

def gui_control_release(self, event):
	try:
		if self.shift_motion_event:
			self.window_canvas.config(cursor="")
			self.window_canvas.unbind('<Motion>',self.shift_motion_event)
			del self.shift_motion_event
			del self.shift_motion_position
	except:
		pass