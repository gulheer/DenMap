from tkinter import Toplevel, Label, Message, CENTER, LEFT
import numpy as np
from PIL import ImageTk

def gui_about(self):
	about_me_message = [
		"DenMap \u00A9 "+self.version_text+", "+self.year_text+", was created by Karl Tassenberg (KTassenberg@hotmail.com), Bogdan Nenchev (Bogdan.Nenchev@gmail.com), Joel Strickland (Joel_Strickland@hotmail.co.uk), and Samuel Perry (SamJP823@gmail.com).",
		"All rights reserved. Any reproduction of part or all of the contents in any form is prohibited without our express written permission.",
		"For the DenMap user guide, future updates/upgrades, please go to www.DenMap.co.uk.",
		"If you would like to use DenMap for commercial purposes, then please contact the authors.",
		"If you would like to use DenMap for academic purposes, then please site the following papers:",
		"For feature extraction:\nNenchev, B., Strickland, J., Tassenberg, K., Perry, S., Gill, S. and Dong, H., 2020. Automatic Recognition of Dendritic Solidification Structures: DenMap. Journal of Imaging, 6(4), p.19.",
		"For GUI/automation:\nTassenberg, K., Nenchev, B., Strickland, J., Perry, S. and Weston, D., 2020. DenMap Single Crystal Solidification Structure Feature Extraction: Automation and Application, Materials Characterization.",
		"For characterisation:\nStrickland, J., Nenchev, B., Perry, S., Tassenberg, K., Gill, S., Panwisawas, C., Dong, H., D'Souza, N. and Irwin, S., 2020. On the nature of hexagonality within the solidification structure of single crystal alloys: Mechanisms and applications. Acta Materialia, 200, pp.417-431."
	]
	about_window = Toplevel()
	#about_window.geometry("250x100+"+str(window.winfo_screenwidth()//2-125)+"+"+str((screen_height-about_window.winfo_reqheight())//2))
	x_size = 485
	y_per_msg = np.ceil(np.array(list(map(len,about_me_message)))*6/(x_size-20))
	extra_line = [about_me_message[i].count("\n") for i in range(len(about_me_message))]
	y_size = int(65+len(about_me_message)*14.5+np.sum(y_per_msg)*14.5+np.sum(extra_line)*14.5)
	#y_size = 300
	about_window.geometry(str(x_size)+"x"+str(y_size)+"+"+str(self.window.winfo_screenwidth()//2-x_size//2)+"+"+str((self.screen_height-about_window.winfo_reqheight())//2-y_size//2))
	about_window.resizable(0,0)
	about_window.title(u"About DenMap \u00A9 "+self.version_text)
	about_window.transient(self.window)
	about_window.grab_set()
	img = ImageTk.PhotoImage(self.impact_logo)
	uol = ImageTk.PhotoImage(self.uol_logo)
	logo = Label(about_window,image=img,justify=CENTER)
	#logo.pack()
	logo.grid(row=0, column=1)
	uol_label = Label(about_window,image=uol,justify=CENTER)
	uol_label.grid(row=0, column=0)
	label = Message(about_window, text="\n\n".join(about_me_message),
									width=x_size-20,justify=LEFT)
	label.grid(row=2,column=0,columnspan=2)
	self.window.wait_window(about_window)