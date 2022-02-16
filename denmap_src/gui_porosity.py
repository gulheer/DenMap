from tkinter import Toplevel, Canvas, Menu, filedialog, BooleanVar, NW, BOTH, YES
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import cv2

def gui_calculate_display_porosity_voronoi(self):
	if not self.check_analysis_data():
		return
	porosity_voronoi_window = Toplevel()
	porosity_voronoi_window.withdraw()
	def porosity_voronoi_close():
		porosity_voronoi_window.destroy()
		if getattr(self,'porosity_voronoi_display_image'):
			del self.porosity_voronoi_display_image
		plt.close('all')
	porosity_voronoi_window.protocol("WM_DELETE_WINDOW", porosity_voronoi_close)
	porosity_voronoi_window.title("Porous Voronoi Overlay Image")
	porosity_voronoi_window.transient(self.window)
	porosity_voronoi_window.grab_set()
	scale = min(self.screen_width/self.main_image.shape[1],self.screen_height/self.main_image.shape[0])*0.9
	im_h = int(self.main_image.shape[0]*scale)
	im_w = int(self.main_image.shape[1]*scale)
	porosity_voronoi_window.geometry(f"{im_w}x{im_h}+{int((self.screen_width-im_w)//2)}+{self.window_y_pos}")
	porosity_voronoi_canvas = Canvas(porosity_voronoi_window,bd=-2)
	porosity_voronoi_canvas.pack(fill=BOTH, expand=YES)
	porosity_voronoi_image = self.analysis_data.generate_porosity_voronoi_image(self.main_image,show_points=False,color_bar=True)
	self.porosity_voronoi_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(porosity_voronoi_image[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
	porosity_voronoi_canvas.create_image(0,0,anchor=NW,image=self.porosity_voronoi_display_image)
	porosity_voronoi_window.update()
	color_bar = BooleanVar()
	porosity_voronoi_window.resizable(0,0)
	def porosity_voronoi_right_click(event):
		try:
			porosity_voronoi_popup.tk_popup(event.x_root, event.y_root)
			self.pop_x = event.x
			self.pop_y = event.y
		finally:
			self.popup.grab_release()
	def porosity_voronoi_save_plot():
		img_save_name = filedialog.asksaveasfilename(title="Select file to save the porous voronoi plot",filetypes=[("Portable Network Graphics",".png"),("Tagged Image File Format",".tiff")],defaultextension=".png")
		if img_save_name:
			if color_bar.get():
				cv2.imwrite(img_save_name,porosity_voronoi_image[:, :, ::-1])
			else:
				cv2.imwrite(img_save_name,porosity_voronoi_image[:self.main_image.shape[0],:self.main_image.shape[1], ::-1])
	def porosity_voronoi_color_bar():
		nonlocal porosity_voronoi_canvas, porosity_voronoi_window
		if color_bar.get():
			scale_cb = min(self.screen_width/porosity_voronoi_image.shape[1],self.screen_height/porosity_voronoi_image.shape[0])*0.9
			im_h_cb = int(porosity_voronoi_image.shape[0]*scale_cb)
			im_w_cb = int(porosity_voronoi_image.shape[1]*scale_cb)
			self.porosity_voronoi_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(porosity_voronoi_image,(im_w_cb,im_h_cb),interpolation = cv2.INTER_CUBIC)))
			porosity_voronoi_canvas.delete("all")
			porosity_voronoi_canvas.create_image(0,0,anchor=NW,image=self.porosity_voronoi_display_image)
			porosity_voronoi_window.geometry(f"{im_w_cb}x{im_h_cb}+{(self.screen_width-im_w_cb)//2}+{self.window_y_pos}")
			porosity_voronoi_window.update()
		else:
			self.porosity_voronoi_display_image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(porosity_voronoi_image[:self.main_image.shape[0],:self.main_image.shape[1]],(im_w,im_h),interpolation = cv2.INTER_CUBIC)))
			porosity_voronoi_canvas.delete("all")
			porosity_voronoi_canvas.create_image(0,0,anchor=NW,image=self.porosity_voronoi_display_image)
			porosity_voronoi_window.geometry(f"{im_w}x{im_h}+{(self.screen_width-im_w)//2}+{self.window_y_pos}")
			porosity_voronoi_window.update()
	porosity_voronoi_window.bind("<Button-3>",porosity_voronoi_right_click)
	porosity_voronoi_popup = Menu(porosity_voronoi_window,tearoff=0)
	porosity_voronoi_popup.add_command(label="Save plot",command=porosity_voronoi_save_plot)
	porosity_voronoi_popup.add_checkbutton(label="Color bar",command=porosity_voronoi_color_bar, offvalue=0, onvalue=1, variable=color_bar)
	porosity_voronoi_window.update()
	porosity_voronoi_window.deiconify()