from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.path as mpltPath
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2, naturalneighbor, concave_hull
from scipy.ndimage import interpolation
from scipy.spatial import cKDTree

def create_colourbar(image,minima,maxima,cmap="jet"):
	norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
	a = np.array([(minima,maxima)])
	f = plt.figure(figsize=(0.58,3.77))
	i = plt.imshow(a, cmap=cmap)
	plt.gca().set_visible(False)
	f.tight_layout(pad=0)
	cax = plt.axes([0.1, 0.02, 0.35, 0.93])
	plt.colorbar(cax=cax)
	new_dpi = image.shape[0]/3.77
	f.dpi = new_dpi
	f.canvas.draw()
	max_len = max([len(i.get_text()) for i in cax.get_yticklabels()])
	f.set_size_inches((0.58+(max_len-1)*0.12,3.77))
	f.canvas.draw()
	data = np.array(f.canvas.renderer.buffer_rgba())[:,:,0:3]
	plt.close('all')
	return data

def create_colourbar_std(image,std,mean,minima,maxima,cmap="jet"):
	#norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
	#mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
	a = np.array([((minima-mean)/std,(maxima-mean)/std)])
	f = plt.figure(figsize=(0.58,3.77))
	i = plt.imshow(a, cmap=cmap)
	plt.gca().set_visible(False)
	f.tight_layout(pad=0)
	cax = plt.axes([0.1, 0.02, 0.35, 0.93])
	plt.colorbar(cax=cax)
	new_dpi = image.shape[0]/3.77
	f.dpi = new_dpi
	f.canvas.draw()
	new_labels = cax.get_yticks()
	int_labels = new_labels.copy()
	new_labels = new_labels.astype(str)
	new_labels[int_labels == int_labels.astype(int)] = int_labels[int_labels == int_labels.astype(int)].astype(int).astype(str)
	max_len = max([len(i)+1 for i in new_labels])
	new_labels = np.array([l+"σ" for l in new_labels])
	new_labels[int_labels == 0] = "0"
	cax.set_yticklabels(new_labels)
	f.set_size_inches((0.58+(max_len-1)*0.12,3.77))
	f.canvas.draw()
	data = np.array(f.canvas.renderer.buffer_rgba())[:,:,0:3]
	plt.close(f)
	return data

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def PolygonSort(corners):
	# calculate centroid of the polygon
	n = len(corners) # of corners
	cx = float(sum(x for x, y in corners)) / n
	cy = float(sum(y for x, y in corners)) / n
	# create a new list of corners which includes angles
	cornersWithAngles = []
	for x, y in corners:
		an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
		cornersWithAngles.append((x, y, an))
	# sort it using the angles
	cornersWithAngles.sort(key = lambda tup: tup[2])
	# return the sorted corners w/ angles removed
	return cornersWithAngles

class stats:
	def std_total(self):
		spacings = []
		for v in range(3,self.max_neigh):
			for i in np.where(self.neighs[:,1] == v)[0]:
				p = self.points[i]
				spacings += [np.sqrt(np.sum((self.points[self.neighs[i,0]]-p)**2,axis=1))]
		return np.std(np.concatenate(spacings))

	def calc_K_and_pdas_ratio(self):
		K_vals = []
		a_pdas = []
		for v in range(3,self.max_neigh):
			N_k = []
			for i in np.where(self.neighs[:,1] == v)[0]:
				p = self.points[i]
				n = self.points[self.neighs[i,0]]
				poly = mpltPath.Path(n)
				if poly.contains_point(p):
					n = np.array(PolygonSort(n))[:,:2]
					area = PolygonArea(n)/len(n)
					a = np.sum(np.sqrt(np.sum((np.roll(n,-1,axis=0)-n)**2,axis=1)))/len(n)
				else:
					P_sort = PolygonSort(np.insert(n,len(n),p,axis=0))
					area = PolygonArea(P_sort)/(len(n)+1)
					n_sort = np.array(P_sort)[:,0:2]
					a = np.sum(np.sqrt(np.sum((np.roll(n_sort,-1,axis=0)-n_sort)**2,axis=1)))/(len(n)+1)
				m_space = np.mean(np.sqrt(np.sum((n-p)**2,axis=1)))
				N_k += [m_space/np.sqrt(area)]
				a_pdas += [a/m_space]
			K_vals += [N_k]
		self.K = np.array(K_vals)
		self.a_pdas_ratio = np.array(a_pdas)
		self.K_mean = []
		self.K_std = []
		for i in self.K:
			self.K_mean += [np.mean(i)]
			self.K_std += [np.std(i)]
		self.a_pdas_mean = []
		self.a_pdas_std = []
		self.a_pdas_error = []
		for i in self.a_pdas_ratio:
			self.a_pdas_mean += [np.mean(i)]
			self.a_pdas_std += [np.std(i)]
			self.a_pdas_error += [[np.mean(i)-np.min(i),np.max(i)-np.mean(i)]]
	
	def __init__(self,centres,scale = 1,boundary=False,image = None):
		self.vor = Voronoi(centres,incremental=True)
		self.verts = self.vor.vertices #Points where the Voronoi regions are
		self.reg = self.vor.regions #List of regions relating to verticies
		self.point_reg = self.vor.point_region #List of points to regions / point_reg[point_num] = region_num
		self.points = self.vor.points
		self.ridge_vertices = np.array(self.vor.ridge_vertices)
		self.err_cent = []
		self.nn_interpolated_values = False
		self.nn_truth_grid = False
		self.pdas_strain = False
		self.pixel_scale = scale
		self.porosity = False
		#coords = self.points[ConvexHull(points=self.points).vertices]
		if type(boundary) == bool and type(image) == type(None):
			self.boundary_coords = concave_hull.compute(self.points, 50).astype(int)
			poly = mpltPath.Path(self.boundary_coords)
			err_vert = np.where(poly.contains_points(self.verts) == False)[0]
			#opt = alphashape.optimizealpha(self.points)*0.9
			#self.boundary_coords = np.array(list(zip(*alphashape.alphashape(self.points,opt).exterior.coords.xy)),dtype=int)
		#self.boundary_indicies = np.array([np.where((self.points[:,0] == i[0])&(self.points[:,1] == i[1]))[0][0] for i in coords])
		elif type(boundary) == bool and type(image) != type(None):
			try:
				self.boundary_coords, self.sample_inner_bounds, self.sample_holes, self.sample_hole_areas, self.sample_hole_points = self.calc_porosity_and_boundaries(image)
				self.porosity = True
				poly = mpltPath.Path(self.boundary_coords)
				err_vert = np.where(poly.contains_points(self.verts) == False)[0]
				for bounds in self.sample_inner_bounds:
					poly = mpltPath.Path(np.concatenate(bounds))
					err_vert = np.append(err_vert,np.where(poly.contains_points(self.verts) == True)[0])
				err_vert = np.unique(err_vert).astype(int)
			except:
				print("Error in finding image boundary, falling back to boundary by points.")
				self.porosity = False
				self.boundary_coords = concave_hull.compute(self.points, 50).astype(int)
				poly = mpltPath.Path(self.boundary_coords)
				err_vert = np.where(poly.contains_points(self.verts) == False)[0]
		else:
			poly = mpltPath.Path(boundary)
			err_vert = np.where(poly.contains_points(self.verts) == False)[0]

		self.err_cent = np.array([i for i in range(len(self.points)) if np.any(np.in1d(err_vert,self.reg[self.point_reg[i]]))])
		self.boundary_centres = [i for i in range(len(self.points)) if -1 in self.reg[self.point_reg[i]] or np.any(np.in1d(self.err_cent,i))]

		self.neighs = []
		for i,p_r in enumerate(self.point_reg):
			r = self.reg[p_r]
			if r == -1 or np.any(np.in1d(self.err_cent,i)):
				self.neighs += [[np.array([]),0]]
				continue
			roll_r = np.roll(r,-1)
			ridges = []
			for v in range(len(r)):
				rid = np.where((self.ridge_vertices[:,1] == r[v])&(self.ridge_vertices[:,0] == roll_r[v]))[0]
				if len(rid) > 0:
					ridges += [rid[0]]
				else:
					ridges += [np.where((self.ridge_vertices[:,0] == r[v])&(self.ridge_vertices[:,1] == roll_r[v]))[0][0]]
			self.neighs += [[self.vor.ridge_points[ridges][np.where(self.vor.ridge_points[ridges] != i)],len(ridges)]]
		self.neighs = np.array(self.neighs)
		self.old_neighs = self.neighs.copy()
		self.max_neigh = max(self.neighs[:,1])+1
		K_SLS = 2*np.sqrt(np.tan(np.pi/np.arange(3,self.max_neigh)))/(2*np.sin(np.pi/np.arange(3,self.max_neigh)))
		K_SLS = np.insert(K_SLS,0,np.zeros(3))
		while True:
			update = 0
			self.max_neigh = max(self.neighs[:,1])+1
			std = self.std_total()
			for i,p in enumerate(self.points):
				r = self.reg[self.point_reg[i]]
				if not -1 in r and not np.any(np.in1d(self.err_cent,i)):
					new_neigh = self.neighs[i][0]
					spacing = np.sqrt(np.sum((self.points[new_neigh]-p)**2,axis=1))
					N = len(spacing)
					R_SLPS = np.mean(spacing) + K_SLS[N]*std
					while True:
						to_remove = np.where(spacing > R_SLPS)[0]
						if len(to_remove):
							if len(spacing)-len(to_remove) > 2:
								spacing = np.delete(spacing,to_remove)
								new_neigh = np.delete(new_neigh,to_remove)
								N = len(spacing)
								R_SLPS = np.mean(spacing) + K_SLS[N]*std
							elif len(spacing) > 3:
								to_remove = np.argsort(spacing)[3-len(spacing):]
								spacing = np.delete(spacing,to_remove)
								new_neigh = np.delete(new_neigh,to_remove)
								N = len(spacing)
								R_SLPS = np.mean(spacing) + K_SLS[N]*std
							else:
								break
						else:
							break
					if len(spacing) < len(self.neighs[i][0]):
						update += 1
						self.neighs[i] = np.array([np.array(new_neigh),len(new_neigh)])
			if update == 0:
				break
		self.calc_K_and_pdas_ratio()
		self.spacings = []
		self.pdas_range = []
		for v in range(3,self.max_neigh):
			for i in np.where(self.neighs[:,1] == v)[0]:
				p = self.points[i]
				self.spacings += [np.sqrt(np.sum((self.points[self.neighs[i,0]]-p)**2,axis=1))]
				self.pdas_range += [np.max(self.spacings[-1])/np.min(self.spacings[-1])]
		self.spacings = np.array(self.spacings)*scale
		self.pdas_range = np.array(self.pdas_range) 
		self.pdas_mean = np.mean(np.concatenate(self.spacings))
		self.pdas_std = np.std(np.concatenate(self.spacings))
		self.K_SLS = K_SLS
		self.boundary_analysis()
		if self.porosity:
			self.perform_porosity_analysis()
	
	def spacing_outlier(self, low_bound, high_bound):
		pts_ind = np.concatenate([np.where(self.neighs[:,1] == v)[0] for v in range(3,self.max_neigh)])
		low_outliers = [i for i in range(len(self.spacings)) if np.any(self.spacings[i] <= low_bound)]
		high_outliers = [i for i in range(len(self.spacings)) if np.any(self.spacings[i] >= high_bound)]
		outliers_list = np.unique(np.concatenate([low_outliers,high_outliers])).astype(int)
		return self.points[pts_ind[outliers_list]].astype(int)



	def plot_shape_voronoi(self,axes,image=None,color_bar = True, show_points = True):
		shape = np.array(self.neighs[:,1])
		minima = 2.5
		maxima = 9.5
		voronoi_colour_array = np.array([
			[76,66,128],
			[126,137,233],
			[96,162,213],
			[83,187,183],
			[153,198,129],
			[218,179,105],
			[220,220,81]
		])/255
		voronoi_cmap = mpl.colors.ListedColormap(voronoi_colour_array,"voronoi")
		norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
		mapper = cm.ScalarMappable(norm=norm, cmap=voronoi_cmap)
		if type(image) != type(None):
			axes.imshow(image,cmap="gray",vmin=0,vmax=255)
		for r in range(len(self.point_reg)):
			region = self.reg[self.point_reg[r]]
			if not -1 in region and not np.any(np.in1d(self.err_cent,r)):
				polygon = [self.vor.vertices[i] for i in region]
				axes.fill(*zip(*polygon), color=mapper.to_rgba(shape[r],alpha=0.7))
				axes.plot(*np.insert(polygon,0,polygon[-1],axis=0).T,color="black",linewidth=0.5)
		self.colorbar = None
		self.mapper = mapper
		if color_bar:
			divider = make_axes_locatable(axes)
			cax = divider.append_axes("right", size="5%", pad=0.05)
			self.colorbar = plt.colorbar(mapper,ax=axes,cax=cax)
		if show_points:
			axes.scatter(*self.points.T,marker="+",color="white",linewidth=1,zorder=3)
		return True

	def generate_voronoi_image(self,image,color_bar = True, show_points = True):
		shape = np.array(self.neighs[:,1])
		minima = 2.5
		maxima = 9.5
		voronoi_colour_array = np.array([
			[76,66,128],
			[126,137,233],
			[96,162,213],
			[83,187,183],
			[153,198,129],
			[218,179,105],
			[220,220,81]
		])/255
		voronoi_cmap = mpl.colors.ListedColormap(voronoi_colour_array,"voronoi")
		norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
		mapper = cm.ScalarMappable(norm=norm, cmap=voronoi_cmap)
		if color_bar:
			a = np.array([mapper.get_clim()])
			f = plt.figure(figsize=(0.58,3.77))
			i = plt.imshow(a, cmap=mapper.cmap)
			plt.gca().set_visible(False)
			f.tight_layout(pad=0)
			cax = plt.axes([0.06, 0.02, 0.35, 0.96])
			plt.colorbar(cax=cax)
			new_dpi = image.shape[0]/3.77
			f.dpi = new_dpi
			f.canvas.draw()
			data = np.array(f.canvas.renderer.buffer_rgba())[:,:,0:3]
			plt.close('all')
			image_overlay = np.ones((image.shape[0],image.shape[1]+data.shape[1],3),dtype=np.uint8)*255
			if len(image.shape) < 3:
				image_overlay[:image.shape[0],:image.shape[1],:] = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
			else:
				image_overlay[:image.shape[0],:image.shape[1],:] = image
			image_overlay[:min(data.shape[0],image_overlay.shape[0]),-data.shape[1]:,:] = data[:min(data.shape[0],image_overlay.shape[0]),:,:]
		else:
			image_overlay = image.copy()
			if len(image_overlay.shape) < 3:
				image_overlay = cv2.cvtColor(image_overlay,cv2.COLOR_GRAY2RGB)
		if self.porosity:
			voronoi_overlay = np.ones(image_overlay.shape)
			voronoi_overlay[:,:] = [255,0,0]
			voronoi_overlay = voronoi_overlay.astype(image_overlay.dtype)
		else:
			voronoi_overlay = image_overlay.copy()
		poly_list = []
		colours = mapper.to_rgba(shape.astype(int))[:,:3]
		for r in range(len(self.point_reg)):
			region = self.reg[self.point_reg[r]]
			if not -1 in region and not np.any(np.in1d(self.err_cent,r)):
				polygon = np.array([list(np.int64(self.verts[i])) for i in region],int)
				poly_list += [polygon]
				cv2.fillPoly(voronoi_overlay,[polygon],colours[r]*255)
		line_scale = max(1,int(max(image.shape[0]/1080,image.shape[1]/1920)))
		if self.porosity:
			if len(self.sample_holes) > 0:
				cv2.drawContours(voronoi_overlay,self.sample_holes,-1,0,-1)
			if len(self.sample_inner_bounds) > 0:
				cv2.drawContours(voronoi_overlay,self.sample_inner_bounds,-1,0,-1)
			np_v = np.array(self.verts)
			poly_draw = [np_v[self.reg[i]].astype(int) for i in range(len(self.reg)) if -1 not in self.reg[i]]
			poly_draw += [np_v[np.array(self.reg[i])[np.array(self.reg[i]) != -1]].astype(int) for i in range(len(self.reg)) if -1 in self.reg[i]]
			cv2.polylines(voronoi_overlay,np.array(poly_draw),True,(0,0,0),line_scale)
			mask = np.zeros(image_overlay.shape,np.uint8)
			cv2.drawContours(mask,[self.boundary_coords],-1,(1,1,1),-1)
			voronoi_overlay = mask*voronoi_overlay + image_overlay*(1-mask)
		else:
			cv2.polylines(voronoi_overlay,poly_list,True,(0,0,0),line_scale)
		if show_points:
			mS = 10*line_scale
			#thick = int(2.5*line_scale)
			for x,y in self.points.astype(int):
				#cv2.drawMarker(voronoi_overlay,(x,y),color=(255,255,255),markerType=cv2.MARKER_CROSS,markerSize = mS,thickness=thick)
				cv2.circle(voronoi_overlay,(x,y),radius = mS,color=(255,255,255),thickness=-1)
		return cv2.addWeighted(image_overlay,0.3,voronoi_overlay,0.7,0)

	def generate_porosity_voronoi_image(self,image,color_bar = True, show_points = True):
		if not self.porosity:
			return 0
		shape = np.array([self.porosity_core_N_holes[self.sample_hole_unique ==i][0] if i in self.sample_hole_unique else 0 for i in range(len(self.points))])
		minima = 0
		maxima = np.max(self.porosity_core_N_holes)
		norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
		mapper = cm.ScalarMappable(norm=norm, cmap="jet")
		if color_bar:
			a = np.array([mapper.get_clim()])
			f = plt.figure(figsize=(0.58,3.77))
			i = plt.imshow(a, cmap=mapper.cmap)
			plt.gca().set_visible(False)
			f.tight_layout(pad=0)
			cax = plt.axes([0.06, 0.02, 0.35, 0.96])
			plt.colorbar(cax=cax)
			new_dpi = image.shape[0]/3.77
			f.dpi = new_dpi
			f.canvas.draw()
			data = np.array(f.canvas.renderer.buffer_rgba())[:,:,0:3]
			plt.close('all')
			image_overlay = np.ones((image.shape[0],image.shape[1]+data.shape[1],3),dtype=np.uint8)*255
			if len(image.shape) < 3:
				image_overlay[:image.shape[0],:image.shape[1],:] = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
			else:
				image_overlay[:image.shape[0],:image.shape[1],:] = image
			image_overlay[:min(data.shape[0],image_overlay.shape[0]),-data.shape[1]:,:] = data[:min(data.shape[0],image_overlay.shape[0]),:,:]
		else:
			image_overlay = image.copy()
			if len(image_overlay.shape) < 3:
				image_overlay = cv2.cvtColor(image_overlay,cv2.COLOR_GRAY2RGB)
		voronoi_overlay = image_overlay.copy()
		colours = mapper.to_rgba(shape.astype(int))[:,:3]
		for r in range(len(self.point_reg)):
			region = np.array(self.reg[self.point_reg[r]])
			if not -1 in region and not np.any(np.in1d(self.err_cent,r)):
				polygon = self.verts[region].astype(int)
				cv2.fillPoly(voronoi_overlay,[polygon],colours[r]*255)
			else:
				invalid_v = np.where(region == -1)[0]
				polygon = self.verts[region[region != -1]].astype(int)
				cv2.fillPoly(voronoi_overlay,[polygon],colours[r]*255)
		line_scale = max(1,int(max(image.shape[0]/1080,image.shape[1]/1920)))

		if len(self.sample_holes) > 0:
			cv2.drawContours(voronoi_overlay,self.sample_holes,-1,0,-1)
		if len(self.sample_inner_bounds) > 0:
			cv2.drawContours(voronoi_overlay,self.sample_inner_bounds,-1,0,-1)
		np_v = np.array(self.verts)
		poly_draw = [np_v[self.reg[i]].astype(int) for i in range(len(self.reg)) if -1 not in self.reg[i]]
		poly_draw += [np_v[np.array(self.reg[i])[np.array(self.reg[i]) != -1]].astype(int) for i in range(len(self.reg)) if -1 in self.reg[i]]
		cv2.polylines(voronoi_overlay,np.array(poly_draw),True,(0,0,0),line_scale)
		mask = np.zeros(image_overlay.shape,np.uint8)
		cv2.drawContours(mask,[self.boundary_coords],-1,(1,1,1),-1)
		voronoi_overlay = mask*voronoi_overlay + image_overlay*(1-mask)

		if show_points:
			mS = 10*line_scale
			for x,y in self.points.astype(int):
				cv2.circle(voronoi_overlay,(x,y),radius = mS,color=(255,255,255),thickness=-1)
		return cv2.addWeighted(image_overlay,0.3,voronoi_overlay,0.7,0)

	def boundary_analysis(self):
		self.boundary_spacings = []
		self.boundary_range = []
		max_spacing = np.amax(np.concatenate(self.spacings))
		for i in self.boundary_centres:
			r = np.array(self.reg[self.point_reg[i]])
			r = r[r != -1]
			roll_r = np.roll(r,-1)
			ridges = []
			for v in range(len(r)):
				rid = np.where((self.ridge_vertices[:,1] == r[v])&(self.ridge_vertices[:,0] == roll_r[v]))[0]
				if len(rid) > 0:
					ridges += [rid[0]]
				else:
					ridge = np.where((self.ridge_vertices[:,0] == r[v])&(self.ridge_vertices[:,1] == roll_r[v]))[0]
					if len(ridge) > 0:
						ridges += [ridge[0]]
			if len(ridges) > 0:
				neigh_pts = self.points[self.vor.ridge_points[ridges][self.vor.ridge_points[ridges] != i]]
				spacing = self.pixel_scale*np.sqrt(np.sum((neigh_pts-self.points[i])**2,axis=1))
				spacing = spacing[spacing <= max_spacing]
				if len(spacing) > 0:
					self.boundary_spacings += [spacing]
					self.boundary_range += [np.max(spacing)/np.min(spacing)]
				else:
					self.boundary_spacings += [[]]
					self.boundary_range += [0]
			else:
				self.boundary_spacings += [[]]
				self.boundary_range += [0]
		self.boundary_spacings = np.array(self.boundary_spacings)
		self.boundary_range = np.array(self.boundary_range)
		boundary_pts = self.boundary_coords
		if self.porosity:
			for b_pts in self.sample_inner_bounds:
				boundary_pts = np.append(boundary_pts,np.concatenate(b_pts),axis=0)
		the_tree = cKDTree(boundary_pts)
		_, b_ind = the_tree.query(self.points[self.boundary_centres])
		fake_voronoi_points = boundary_pts[b_ind]*2-self.points[self.boundary_centres]
		fake_voronoi = Voronoi(np.append(self.points,fake_voronoi_points,axis=0))
		self.boundary_voronoi_areas = self.pixel_scale**2*np.array([cv2.contourArea(np.round(fake_voronoi.vertices[fake_voronoi.regions[fake_voronoi.point_region[i]]]).astype(int)) for i in range(len(self.points))])

	def calc_pdas_strain(self):
		sp_N = np.array([len(s) for s in self.spacings])
		sp_6_mean = np.mean(np.concatenate(self.spacings[np.where(sp_N == 6)[0]]))
		max_spacing = np.amax(np.concatenate(self.spacings))
		pdas_strain = []
		for i in range(len(self.points)):
			if -1 in self.reg[self.point_reg[i]] or np.any(np.in1d(self.err_cent,i)):
				ind = self.boundary_centres.index(i)
				pd_s = self.boundary_spacings[ind]
				if len(pd_s) == 0:
					continue
				pd_s = pd_s[pd_s < max_spacing]
				if len(pd_s) > 0:
					pd_s = (np.mean(pd_s)-sp_6_mean)/sp_6_mean
					pdas_strain += [[self.points[i][0],self.points[i][1],0,pd_s]]
			else:
				neigh_pts = self.points[self.neighs[i][0]]
				pd_s = (self.pixel_scale*np.mean(np.sqrt(np.sum((neigh_pts-self.points[i])**2,axis=1)))-sp_6_mean)/sp_6_mean
				pdas_strain += [[self.points[i][0],self.points[i][1],0,pd_s]]
		self.pdas_strain = np.array(pdas_strain)
		return self.pdas_strain

	def calc_heat_map(self,image):
		if type(self.pdas_strain) == bool:
			pdas_strain = self.calc_pdas_strain()
		else:
			pdas_strain = self.pdas_strain
		max_y, max_x = image.shape[:2]
		heat_res_factor = 20
		grid_ranges = [[0, max_x, heat_res_factor], [0, max_y, heat_res_factor], [0, 1, 1]]
		nn_interpolated_values = naturalneighbor.griddata(pdas_strain[:,:3], pdas_strain[:,3], grid_ranges)
		nn_interpolated_values = np.concatenate(nn_interpolated_values,axis=1)
		nn_interpolated_values = interpolation.zoom(nn_interpolated_values,max(max_y/nn_interpolated_values.shape[0],max_x/nn_interpolated_values.shape[1]))
		nn_interpolated_values = nn_interpolated_values[:max_y,:max_x]
		truth_grid = np.zeros(nn_interpolated_values.shape)
		cv2.fillPoly(truth_grid,[self.boundary_coords],1)
		if self.porosity:
			if len(self.sample_holes) > 0:
				cv2.drawContours(truth_grid,self.sample_holes,-1,0,-1)
			if len(self.sample_inner_bounds) > 0:
				cv2.drawContours(truth_grid,self.sample_inner_bounds,-1,0,-1)
		truth_grid[truth_grid == 0] = np.nan
		self.nn_interpolated_values = nn_interpolated_values
		self.nn_truth_grid = truth_grid
		return nn_interpolated_values,truth_grid

	def calc_fit_surface(self,image):
		if type(self.pdas_strain) == bool:
			pdas_strain = self.calc_pdas_strain()
		else:
			pdas_strain = self.pdas_strain
		max_y, max_x = image.shape[:2]
		heat_res_factor = 20
		grid_ranges = [[0, max_x, heat_res_factor], [0, max_y, heat_res_factor], [0, 1, 1]]
		nn_interpolated_values = naturalneighbor.griddata(pdas_strain[:,:3], pdas_strain[:,3], grid_ranges)
		nn_interpolated_values = np.concatenate(nn_interpolated_values,axis=1)
		nn_interpolated_values = interpolation.zoom(nn_interpolated_values,max(max_y/nn_interpolated_values.shape[0],max_x/nn_interpolated_values.shape[1]))
		nn_interpolated_values = nn_interpolated_values[:max_y,:max_x]
		truth_grid = np.zeros(nn_interpolated_values.shape)
		cv2.fillPoly(truth_grid,[self.boundary_coords],1)
		if self.porosity:
			if len(self.sample_holes) > 0:
				cv2.drawContours(truth_grid,self.sample_holes,-1,0,-1)
			if len(self.sample_inner_bounds) > 0:
				cv2.drawContours(truth_grid,self.sample_inner_bounds,-1,0,-1)
		truth_grid[truth_grid == 0] = np.nan
		self.nn_interpolated_values = nn_interpolated_values
		self.nn_truth_grid = truth_grid
		return nn_interpolated_values,truth_grid

	def calc_porosity_and_boundaries(self,image):
		im = image.astype(np.uint8) if len(image.shape) < 3 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		sample_contours,sample_hierarchy = cv2.findContours(cv2.GaussianBlur((im > 6).astype(np.uint8),(3,3),0),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
		sample_contour_areas = np.array(list(map(cv2.contourArea,sample_contours)))
		sample_ind = np.argmax(sample_contour_areas)
		hole_ind = np.where(sample_hierarchy[:,:,3] == sample_ind)[1]
		sample_hole_areas = sample_contour_areas[hole_ind]
		#max_area = (self.pdas_mean/(2*self.pixel_scale))**2
		valid_regions = np.array(self.reg)[[i for i in range(len(self.points)) if -1 not in self.reg[i]]]
		valid_regions = [np.round(self.verts[valid_regions[i]]).astype(int) for i in range(len(valid_regions)) if not True in list(self.verts[valid_regions[i]].flatten() < 0)+list(self.verts[valid_regions[i]][:,0] > image.shape[1])+list(self.verts[valid_regions[i]][:,1] > image.shape[0])]
		max_area = np.mean(list(map(cv2.contourArea,valid_regions)))/4
		sample_holes = np.array(sample_contours)[hole_ind][(sample_hole_areas > 10)&(sample_hole_areas < max_area)]
		sample_inner_bounds = np.array(sample_contours)[hole_ind][(sample_hole_areas >= max_area)]
		sample_outer_bounds = np.concatenate(sample_contours[sample_ind])
		sample_hole_areas = sample_hole_areas[(sample_hole_areas > 10)&(sample_hole_areas < max_area)]
		sample_hole_areas *= self.pixel_scale**2
		moment_list = np.array(list(map(moments_retr,sample_holes)))
		if len(moment_list) > 0:
			moment_list = moment_list[moment_list[:,0] > 0]
			cX = moment_list[:,1]/moment_list[:,0]
			cY = moment_list[:,2]/moment_list[:,0]
			pore_coords = np.array([cX,cY]).astype(int).T
			the_tree = cKDTree(self.points)
			_, sample_hole_points = the_tree.query(pore_coords)
		else:
			cX = []
			cY = []
			pore_coords = np.array([])
			sample_hole_points = np.array([])
		return sample_outer_bounds, sample_inner_bounds, sample_holes, sample_hole_areas, sample_hole_points

	def perform_porosity_analysis(self):
		if len(self.sample_hole_points) > 0:
			sample_neighs = self.neighs[:,1][self.sample_hole_points]
			self.sample_n_holes = np.array([np.sum(sample_neighs == 0)]+[np.sum(sample_neighs == i) for i in range(3,self.max_neigh)])
			sample_hole_unique = np.unique(self.sample_hole_points)
			self.sample_hole_unique = sample_hole_unique
			sample_hole_neighs = self.neighs[:,1][sample_hole_unique]
			sample_hole_unique_area = np.array([np.sum(self.sample_hole_areas[self.sample_hole_points == i]) for i in sample_hole_unique])
			#porosity_normalised_area = sample_hole_unique_area/voronoi_areas[sample_hole_unique]
			porosity_normalised_area = sample_hole_unique_area/self.boundary_voronoi_areas[sample_hole_unique]
			self.porosity_mean_normalised_area = np.mean(porosity_normalised_area)
			self.porosity_std_normalised_area = np.std(porosity_normalised_area)
			self.porosity_mean_normalised_area_per_N = np.array([np.mean(porosity_normalised_area[sample_hole_neighs == 0] if np.sum(sample_hole_neighs == 0) > 0 else 0.0)]+[np.mean(porosity_normalised_area[sample_hole_neighs == i]) if np.sum(sample_hole_neighs == i) > 0 else 0.0 for i in range(3,self.max_neigh)])
			self.porosity_std_normalised_area_per_N = np.array([np.std(porosity_normalised_area[sample_hole_neighs == 0]) if np.sum(sample_hole_neighs == 0) > 0 else 0.0]+[np.std(porosity_normalised_area[sample_hole_neighs == i]) if np.sum(sample_hole_neighs == i) > 0 else 0.0 for i in range(3,self.max_neigh)])

			self.porosity_mean_total_area_per_N = np.array([np.mean(sample_hole_unique_area[sample_hole_neighs == 0]) if np.sum(sample_hole_neighs == 0) > 0 else 0.0]+[np.mean(sample_hole_unique_area[sample_hole_neighs == i]) if np.sum(sample_hole_neighs == i) > 0 else 0.0 for i in range(3,self.max_neigh)])
			self.porosity_std_total_area_per_N = np.array([np.std(sample_hole_unique_area[sample_hole_neighs == 0 ]) if np.sum(sample_hole_neighs == 0) > 0 else 0.0]+[np.std(sample_hole_unique_area[sample_hole_neighs == i]) if np.sum(sample_hole_neighs == i) > 0 else 0.0 for i in range(3,self.max_neigh)])
			self.porosity_mean_total_area = np.mean(sample_hole_unique_area)
			self.porosity_std_total_area = np.std(sample_hole_unique_area)

			self.porosity_core_N_holes = np.array([np.sum(self.sample_hole_points == i) for i in sample_hole_unique])
			self.porosity_mean_n = np.mean(self.porosity_core_N_holes)
			self.porosity_std_n = np.std(self.porosity_core_N_holes)
			self.porosity_mean_n_N = np.array([np.mean(self.porosity_core_N_holes[sample_hole_neighs == 0]) if np.sum(sample_hole_neighs == 0) > 0 else 0.0]+[np.mean(self.porosity_core_N_holes[sample_hole_neighs == i] ) if np.sum(sample_hole_neighs == i) > 0 else 0.0 for i in range(3,self.max_neigh)])
			self.porosity_std_n_N = np.array([np.std(self.porosity_core_N_holes[sample_hole_neighs == 0]) if np.sum(sample_hole_neighs == 0) > 0 else 0.0]+[np.std(self.porosity_core_N_holes[sample_hole_neighs == i]) if np.sum(sample_hole_neighs == i) > 0 else 0.0 for i in range(3,self.max_neigh)  ])


def moments_retr(c):
	M = cv2.moments(c)
	return [M["m00"],M["m10"],M["m01"]]