import numpy as np

def gui_change_original(self, zoom_reset=False):
	if not type(self.original_image) == bool and self.image_checkvariables[0].get():
		if zoom_reset:
			self.zoom_scale = 1
			self.zoom_corner = [0,0]
		self.unset_view_variables()
		self.main_image = self.original_image
		self.image_checkvariables[0].set(1)
		self.zoom_or_update()
	elif not self.image_checkvariables[0].get():
		self.image_checkvariables[0].set(1)


def gui_change_binary(self, zoom_reset=False):
	if not type(self.binary_image) == bool and self.image_checkvariables[1].get():
		if zoom_reset:
			self.zoom_scale = 1
			self.zoom_corner = [0,0]
		self.unset_view_variables()
		self.main_image = self.binary_image
		self.image_checkvariables[1].set(1)
		self.zoom_or_update()
	elif not self.image_checkvariables[1].get():
		self.image_checkvariables[1].set(1)

def gui_change_filt(self, zoom_reset=False):
	if not type(self.new_image) == bool and self.image_checkvariables[2].get():
		if zoom_reset:
			self.zoom_scale = 1
			self.zoom_corner = [0,0]
		self.unset_view_variables()
		self.main_image = self.new_image
		self.image_checkvariables[2].set(1)
		self.zoom_or_update()
	elif not self.image_checkvariables[2].get():
		self.image_checkvariables[2].set(1)

def gui_change_filt_cent(self, zoom_reset=False):
	if not type(self.circle_image) == bool and self.image_checkvariables[3].get():
		if zoom_reset:
			self.zoom_scale = 1
			self.zoom_corner = [0,0]
		self.unset_view_variables()
		self.main_image = np.array(self.circle_image)
		self.image_checkvariables[3].set(1)
		self.zoom_or_update()
	elif not self.image_checkvariables[3].get():
		self.image_checkvariables[3].set(1)