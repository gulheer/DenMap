from tkinter import END

def gui_disable_menu(self):
	last = self.menu_analyse.index(END)+1
	for i in range(1,last):
		self.menu_analyse.entryconfig(i, state="disabled")
	self.menu_export.entryconfig(1,state="disabled")
	self.menu_export.entryconfig(2,state="disabled")
	self.menu_export.entryconfig(3,state="disabled")
	self.menu_image.entryconfig(0,state="disabled")
	last = self.menu_view.index(END)+1
	for i in range(0,last):
		self.menu_view.entryconfig(i, state="disabled")
		self.image_checkvariables[i].set(0)
	last = self.menu_process.index(END)+1
	for i in range(0,last):
		self.menu_process.entryconfig(i, state="disabled")

def unset_view_variables(self):
	for i in self.image_checkvariables:
		i.set(0)

def do_nothing(self):
    pass

def gui_enable_menu(self):
	last = self.menu_analyse.index(END)+1
	for i in range(1,last):
		self.menu_analyse.entryconfig(i, state="normal")
	self.menu_export.entryconfig(1,state="normal")
	self.menu_export.entryconfig(2,state="normal")
	self.menu_export.entryconfig(3,state="normal")
	last = self.menu_process.index(END)+1
	for i in range(0,last):
		self.menu_process.entryconfig(i, state="normal")