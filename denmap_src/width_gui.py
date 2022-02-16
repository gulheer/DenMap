import numpy as np
import cv2, imutils, psutil, skimage.measure
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar, minimize
from itertools import groupby
from operator import itemgetter
import multiprocessing as mp
from functools import partial

import pyfftw
from pyfftw.interfaces.scipy_fftpack import ifft2, fft2, fftshift, ifftshift
pyfftw.config.NUM_THREADS = mp.cpu_count()-1

from reikna.cluda import any_api, dtypes, functions
from reikna.fft import FFT, FFTShift

#Define all functions

def group_conseq(data):
    group = []
    for k, g in groupby(enumerate(data), lambda i : i[0]-i[1]):
        group.append(list(map(itemgetter(1), g)))
    return group

def group_mean(grp):
    m = []
    for i in grp:
        m.append(np.mean(i))
    return m

def group_len(grp):
    l = []
    for i in grp:
        l.append(len(i))
    return l

#def mp_template_initialise(cl):
#	global cont_list
#	cont_list = cl

def group_wid_and_mean(data):
	g_mins = list(map(lambda x: x[0], data))
	g_maxs = list(map(lambda x: x[-1], data))
	return list(map(lambda n: [g_maxs[2*n+1] -g_mins[2*n],(g_maxs[2*n+1] + g_mins[2*n])/2], range(len(data)//2)))

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def group_closest_group(data,mean):
	return np.argmin(list(map(lambda n: abs(mean-data[n][1]), range(len(data)))))

def mp_template_2(v,cont_list):
	widths = -1
	heights = -1
	cont_num = -1
	i = cont_list[v]
	minY,minX,maxY,maxX = min(i[:,0,0]),min(i[:,0,1]),max(i[:,0,0]),max(i[:,0,1])
	i_cont = i[:,0] - (minY,minX)
	side = 0 if (maxY-minY > maxX-minX) else 1
	i_cont = i_cont[ np.argsort(i_cont[:,side]) ,:]
	axis_points = np.split(i_cont[:,(1-side)],np.where(np.diff(i_cont[:,side]) != 0)[0]+1)
	axis_points = list(map(sorted, axis_points)) 
	groups = list(map(consecutive, axis_points)) 
	widths_m = list(map(group_wid_and_mean, groups))
	width_len = np.array(list(map(len, widths_m)))
	zero_positions = np.where(width_len == 0)[0]
	widths_m = np.array(widths_m,dtype=object)[width_len>0]
	width_len = width_len[width_len>0]
	mean_axis = np.mean(np.array(widths_m[width_len == 1].tolist())[:,0,1])
	widths_above_one = (width_len > 1)
	full_ind = np.zeros(len(widths_m),dtype=int)
	if np.sum(widths_above_one) > 1:
		closest_ind = list(map(lambda x: group_closest_group(x,mean_axis), widths_m[widths_above_one]))
		full_ind[widths_above_one] += closest_ind
	len_list = list(map(lambda d: d[0][d[1]][0],zip(widths_m.tolist(),full_ind)))
	list(map(lambda n: len_list.insert(n,0),zero_positions))
	gau_len = gaussian_filter1d(len_list,sigma=3)
	dgau_len = gaussian_filter1d(len_list,sigma=3,order=1)
	max_gau_loc = np.where(gau_len == max(gau_len))[0][0]
	min_loc = np.where(dgau_len == min(dgau_len))[0][0]
	max_loc = np.where(dgau_len == max(dgau_len))[0][0]
	if (max_gau_loc-(min_loc+max_loc)//2 < (min_loc-max_loc)//10) and (min_loc > max_loc) and (dgau_len[max_loc] > 0) and (dgau_len[min_loc] < 0) and (dgau_len[max_loc] > np.mean(dgau_len)+3*np.std(dgau_len)) and (dgau_len[min_loc] < np.mean(dgau_len)-3*np.std(dgau_len)) and (abs(-dgau_len[min_loc] - dgau_len[max_loc]) <= 3):
		#Find width
		loc_w1 = np.where(dgau_len[:max_loc]==0)[0]
		if (not len(loc_w1)):
			return widths,heights,cont_num
		loc_w1 = loc_w1[len(loc_w1)-1]
		loc_w2 = np.where(dgau_len[min_loc:]==0)[0]
		if (not len(loc_w2)): 
			return widths,heights,cont_num
		loc_w2 = loc_w2[0]+min_loc
		w1 = gau_len[loc_w1]
		w2 = gau_len[loc_w2]
		w_h = (w1+w2)//2
		w_v = min_loc-max_loc
		if (w_h/w_v < 1.5 and w_v/w_h < 1.5):
			widths=w_h
			heights=w_v
			cont_num=v
	return widths,heights,cont_num
#template_loop
def mp_template(v,cont_list):
    widths = -1
    heights = -1
    cont_num = -1
    i = cont_list[v]
    minY,minX,maxY,maxX = min(i[:,0,0]),min(i[:,0,1]),max(i[:,0,0]),max(i[:,0,1])
    fill_map = np.zeros((maxX-minX+11,maxY-minY+11),np.uint8)
    i_cont = i[:,0] - (minY-5,minX-5)
    cv2.drawContours(fill_map,[i_cont],-1,1,-1)
    len_list = []
    prev_group = []
    side = 0 if (fill_map.shape[0] > fill_map.shape[1]) else 1
    loop = side*fill_map.shape[1] + int((not side))*fill_map.shape[0]
    for i in range(0,loop):
        wh = np.where(fill_map[:,i]==1)[0] if (side) else np.where(fill_map[i,:]==1)[0]
        if (len(wh)): 
            groups = group_conseq(wh)
            if (not prev_group): 
                len_list.append(max(group_len(groups)))
                prev_group.append(np.mean(groups[0]))
            else:
                means = group_mean(groups)
                means_dif = abs(np.array(means)-np.mean(prev_group))
                wh_grp = np.where(means_dif == min(means_dif))[0][0]
                len_list.append(len(groups[wh_grp]))
                prev_group.append(np.mean(groups[wh_grp]))
        else: len_list.append(0)
    gau_len = gaussian_filter1d(len_list,sigma=3)
    dgau_len = gaussian_filter1d(len_list,sigma=3,order=1)
    max_gau_loc = np.where(gau_len == max(gau_len))[0][0]
    min_loc = np.where(dgau_len == min(dgau_len))[0][0]
    max_loc = np.where(dgau_len == max(dgau_len))[0][0]
    if (max_gau_loc-(min_loc+max_loc)//2 < (min_loc-max_loc)//10) and (min_loc > max_loc) and (dgau_len[max_loc] > 0) and (dgau_len[min_loc] < 0) and (dgau_len[max_loc] > np.mean(dgau_len)+3*np.std(dgau_len)) and (dgau_len[min_loc] < np.mean(dgau_len)-3*np.std(dgau_len)) and (abs(-dgau_len[min_loc] - dgau_len[max_loc]) <= 3):
        #Find width
        loc_w1 = np.where(dgau_len[:max_loc]==0)[0]
        if (not len(loc_w1)):
            return widths,heights,cont_num
        loc_w1 = loc_w1[len(loc_w1)-1]
        loc_w2 = np.where(dgau_len[min_loc:]==0)[0]
        if (not len(loc_w2)): 
            return widths,heights,cont_num
        loc_w2 = loc_w2[0]+min_loc
        w1 = gau_len[loc_w1]
        w2 = gau_len[loc_w2]
        w_h = (w1+w2)//2
        w_v = min_loc-max_loc
        if (w_h/w_v < 1.5 and w_v/w_h < 1.5):
            widths=w_h
            heights=w_v
            cont_num=v
    return widths,heights,cont_num

def min_std(m,cor,image,temp):
    p_centres = width_find_centres(image,cor,m)
    if (len(p_centres) < 2): return 0
    bright = []
    dark = []
    start_x = temp.shape[1]//2
    end_x = temp.shape[1]//2 + temp.shape[1]%2
    start_y = temp.shape[0]//2
    end_y = temp.shape[0]//2 + temp.shape[0]%2
    inv_temp = 1-temp
    for p in p_centres:
        x0 = p[0]-start_x
        x1 = p[0]+end_x
        y0 = p[1]-start_y
        y1 = p[1]+end_y
        if (y0 < 0) or (y1>image.shape[0]) or (x0 < 0) or (x1>image.shape[1]): 
            bright.append(-1)
            dark.append(-1)
            continue
        im = image[y0:y1,x0:x1]
        bright.append(np.mean(im*temp))
        dark.append(np.mean(im*inv_temp))
    temp_size = temp.shape[0]*temp.shape[1]
    dark_size = len(np.where(inv_temp==1)[0])
    bright_size = len(np.where(temp==1)[0])
    d_bright=(np.array(bright)/bright_size-np.array(dark)/dark_size)*temp_size
    outliers = np.where((d_bright<np.mean(d_bright)-2.5*np.std(d_bright))&(d_bright>0))[0]
    outliers = outliers[np.where(np.array(bright)[outliers] >= 0)[0]]
    p_centres = np.delete(p_centres,outliers,0)
    distances = cdist(p_centres,p_centres)
    nearest = []
    for d in distances:
        s = np.argsort(d)
        nearest+=[d[s[1]]]
    n2 = np.array(nearest)[np.where((nearest > np.mean(nearest)-np.std(nearest))&(nearest < np.mean(nearest)+np.std(nearest)))[0] ]
    if len(n2) > 0.5*len(nearest): 
        return np.std(n2)
    else: return np.std(nearest)

def max_std(m,cor,image,temp):
    n = min_std(m,cor,image,temp)
    if (n): return 1/n
    return 1e5

def find_half_area(hist,bins,low,high):
    bin_filter = np.where((bins>low)&(bins<high))[0]
    h = hist[bin_filter]
    cum = np.cumsum(h)
    loc = np.where(cum>cum[len(cum)-1]/2)[0][0]+bin_filter[0]
    return bins[loc]

#Angle in degrees
def rotate_points(points, angle):
    a = angle*np.pi/180
    qx = np.cos(a) * points[:,0] - np.sin(a) * points[:,1]
    qy = np.sin(a) * points[:,0] + np.cos(a) * points[:,1]
    return np.array(list(zip(qx, qy)))

def rot_max_cont(angle,points):
    rotated = rotate_points(points,angle).round().astype(int)
    x_shape = rotated[:,0]
    x_shape = max(x_shape)-min(x_shape)
    y_shape = rotated[:,1]
    y_shape = max(y_shape)-min(y_shape)
    return 1/max(y_shape,x_shape)

def rotate_cont(cont, angle):
    a = angle*np.pi/180
    M = cv2.moments(cont)
    if M['m00'] == 0:
        cx = 0
        cy = 0
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    qx = (np.cos(a) * (cont[:,0,0]-cx) - np.sin(a) * (cont[:,0,1]-cy) + cx).round()
    qy = (np.sin(a) * (cont[:,0,0]-cx) + np.cos(a) * (cont[:,0,1]-cy) + cy).round()
    c_list = []
    for i,j in zip(qx,qy):
        c_list.append([[i,j]])
    return np.array(c_list,dtype=np.int32)

def mp_t_dists(y,cont_a):
    i = cont_a[y]
    minX,minY,maxX,maxY = min(i[:,0,0]),min(i[:,0,1]),max(i[:,0,0]),max(i[:,0,1])
    i_cont = i[:,0] - (minX-5,minY-5)
    s_x = max(i_cont[:,0])-min(i_cont[:,0])
    s_y = max(i_cont[:,1])-min(i_cont[:,1])
    r = s_x/s_y
    if (r < 0.8 or r > 1.2):
        return minimize_scalar(rot_max_cont,args=(i_cont), bounds=[-45,45],method="bounded").x
    else:
        return -100

def perform_fft_cpu(image,r_in = False,r_out = 100):
    if not r_in:
        r_in = calculate_r_in(image)
    filt = 2**np.ceil(np.log2(max(image.shape[0],image.shape[1])))
    x, y = np.meshgrid(np.linspace(np.round(-0.5*image.shape[1]),np.round(0.5*image.shape[1]),image.shape[1]),np.linspace(np.round(-0.5*image.shape[0]),np.round(0.5*image.shape[0]),image.shape[0]))
    factors = x*x+y*y
    x=0
    y=0
    fact_remove = np.where(factors < 1)
    scaleLarge = (2*r_out/filt)**2
    scaleSmall = (2*r_in/filt)**2
    factors = np.multiply((1-np.exp(np.multiply(-factors,scaleLarge))),np.exp(np.multiply(-factors,scaleSmall)))
    if (len(fact_remove[0])):
        factors[fact_remove] = 0
    factors = factors.astype(np.float32)
    image=fftshift(fft2(image))
    image = ifftshift(np.multiply(image,factors))
    factors = 0
    image = ifft2(image)
    image = np.absolute(image - np.amin(image))
    contrast_enhance = 255/(6*np.std(image))
    image *= contrast_enhance
    image += 125 - np.mean(image)
    image = image.round()
    image[np.where(image<0)]=0
    image[np.where(image>255)]=255
    return image.astype(np.uint8)

def gaussian_matrix_fft(image,r_in,r_out):
    filt = 2**np.ceil(np.log2(max(image.shape[0],image.shape[1])))
    x, y = np.meshgrid(np.linspace(np.round(-0.5*image.shape[1]),np.round(0.5*image.shape[1]),image.shape[1]),np.linspace(np.round(-0.5*image.shape[0]),np.round(0.5*image.shape[0]),image.shape[0]))
    factors = x*x+y*y
    del x,y
    fact_remove = np.where(factors < 1)
    scaleLarge = (2*r_out/filt)**2
    scaleSmall = (2*r_in/filt)**2
    factors = np.multiply((1-np.exp(np.multiply(-factors,scaleLarge))),np.exp(np.multiply(-factors,scaleSmall)))
    if (len(fact_remove[0])):
        factors[fact_remove] = 0
    return factors.astype(np.float32)

def attempt_gpu_fft(image, shared_memory=False):
	try:
		if shared_memory:
			raise NameError("Shared Memory: Switching to CPU")
		api = any_api()
		thr = api.Thread.create()
		image_dev = thr.to_device(image.astype(np.complex64))
		gpu_fft = FFT(image_dev,axes=(0,1)).compile(thr)
		gpu_fftshift = FFTShift(image_dev,axes=(0,1)).compile(thr)
		gpu_fft(image_dev, image_dev)
		gpu_fftshift(image_dev,image_dev)
		return image_dev, thr, gpu_fft, gpu_fftshift
	except:
		return False, False, False, False

def inverse_gpu_fft(image_dev, thr, gpu_fft, gpu_fftshift):
    gpu_fftshift(image_dev,image_dev, inverse=True)
    gpu_fft(image_dev, image_dev, inverse=True)
    image = image_dev.get()
    return image

def gpu_multiply_factors(image_dev, thr, factors):
    # dtype = np.complex64
    # program = thr.compile("""
    # KERNEL void multiply_them(
    #     GLOBAL_MEM ${ctype}*dest,
    #     GLOBAL_MEM ${ctype}*a,
    #     GLOBAL_MEM ${ctype}*b,
    #     GLOBAL_MEM int*c)
    # {
    #     for (int i = 0; i < c[0]; ++i)
    #     dest[i] = ${mul}(a[i], b[i]);
    # }
    # """, render_kwds=dict(
    #     ctype=dtypes.ctype(dtype),
    #     mul=functions.mul(dtype, dtype)))
    # multiply_them = program.multiply_them
    # b_dev = thr.to_device(factors.astype(dtype))
    # c_dev = thr.to_device(np.array([factors.shape[0]*factors.shape[1]]).astype(np.int32))
    # multiply_them(image_dev, image_dev, b_dev, c_dev, global_size=1,local_size=1)
    # del b_dev, c_dev
    thr.to_device(np.multiply(image_dev.get(),factors.astype(np.complex64)),dest=image_dev)

def cpu_fft(image):
    return fftshift(fft2(image))

def cpu_multiply_factors(image,factors):
    return ifftshift(np.multiply(image,factors))

def cpu_ifft(image):
    return ifft2(image)

def gpu_shared_memory():
	try:
		api = any_api()
		thr = api.Thread.create()
		img = np.random.rand(2000,2000)
		mem_diff = psutil.virtual_memory().free
		image_dev = thr.to_device(img)
		mem_diff -= psutil.virtual_memory().free
	except:
		return False
	return (mem_diff > 0.7*img.size*img.itemsize)

def perform_image_contrast(image):
    image = np.absolute(image - np.amin(image))
    contrast_enhance = 255/(6*np.std(image))
    image *= contrast_enhance
    image += 125 - np.mean(image)
    image = image.round()
    image[np.where(image<0)]=0
    image[np.where(image>255)]=255
    return image


def calculate_r_in(image):
	hist = cv2.calcHist([image],[0],None,[256],[0,256]).T[0].astype(int)
	hist[-2] += hist[-1]
	hist = hist[:-1]
	bins = np.arange(256)
	#hist, bins = np.histogram(image.reshape(-1),255)
	bins = bins[1:]-0.5
	hist = np.delete(hist,np.arange(0,2))
	bins = np.delete(bins,np.arange(0,2))
	hist = np.delete(hist,np.arange(len(bins)-2,len(bins)))
	bins = np.delete(bins,np.arange(len(bins)-2,len(bins)))
	deconv = deconvolute_max_signal(bins,hist)
	de_mean = np.array([1,4,7,10])
	max_arg = np.argmax(deconv[de_mean])
	#new_im = (image >= deconv[de_mean[max_arg]]-2*deconv[de_mean[max_arg]+1]).astype(int)
	new_im = cv2.threshold(image,np.round(deconv[de_mean[max_arg]]-2*deconv[de_mean[max_arg]+1]).astype(int),1,cv2.THRESH_BINARY)[1]
	#n_lab,lab,lab_stats,centroids = cv2.connectedComponentsWithStats(binary_image)
	_,_,lab_stats,_ = cv2.connectedComponentsWithStats(new_im)
	areas = lab_stats[:,4]
	areas = areas[areas>10]
	areas = areas[areas<np.mean(areas)+2*np.std(areas)]
	return min(max(np.round(np.sqrt(np.mean(areas))*1.2).astype(int),8),50)

def perform_fft(image,shared_memory = False,r_in=False,r_out=100):
	if not r_in:
		r_in = calculate_r_in(image)
	filt = 2**np.ceil(np.log2(max(image.shape[0],image.shape[1])))
	x, y = np.meshgrid(np.linspace(np.round(-0.5*image.shape[1]),np.round(0.5*image.shape[1]),image.shape[1]),np.linspace(np.round(-0.5*image.shape[0]),np.round(0.5*image.shape[0]),image.shape[0]))
	#factors = x*x+y*y
	factors = x*x+y*y
	x=0
	y=0
	fact_remove = np.where(factors < 1)
	scaleLarge = (2*r_out/filt)**2
	scaleSmall = (2*r_in/filt)**2
	#factors = (1-np.exp(-factors*scaleLarge))*np.exp(-factors*scaleSmall)
	factors = np.multiply((1-np.exp(np.multiply(-factors,scaleLarge))),np.exp(np.multiply(-factors,scaleSmall)))
	factors = factors.astype(np.float32)
	mp_factors = False
	fft_img = False
	if (len(fact_remove[0])):
		factors[fact_remove] = 0
	try:
		if shared_memory:
			raise NameError("Shared Memory: Switching to CPU")
		api = any_api()
		thr = api.Thread.create()

		image_dev = thr.to_device(image.astype(np.complex64))
		gpu_fft = FFT(image_dev,axes=(0,1)).compile(thr)
		gpu_fftshift = FFTShift(image_dev,axes=(0,1)).compile(thr)
		gpu_fft(image_dev, image_dev)
		gpu_fftshift(image_dev,image_dev)
		#thr.to_device(image_dev.get()*factors,image_dev)
		
		image = image_dev.get()
		fft_img = True
		image = np.multiply(image,factors)
		mp_factors = True
		#gpu_multiply_factors(image_dev,thr,factors)
		thr.to_device(image,dest=image_dev)
		del factors
		gpu_fftshift(image_dev,image_dev, inverse=True)
		gpu_fft(image_dev, image_dev, inverse=True)
		image = image_dev.get()
	except:
		try:
			if not fft_img:
				image=fftshift(fft2(image))
				image = np.multiply(image,factors)
				del factors
			elif not mp_factors:
				image = np.multiply(image,factors)
				del factors
			image = ifftshift(image)
			image = ifft2(image)
		except:
			return False
	image = np.absolute(image - np.amin(image))
	contrast_enhance = 255/(6*np.std(image))
	image *= contrast_enhance
	image += 125 - np.mean(image)
	image = image.round()
	image[np.where(image<0)]=0
	image[np.where(image>255)]=255
	return image.astype(np.uint8)

def width_contours(binary):
    cont,hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    main_hier = hier[0,:,0]
    cont_list = []
    for i in main_hier:
        if cv2.contourArea(cont[i]) > cv2.arcLength(cont[i], True) and i > -1 and cv2.contourArea(cont[i]) > 5000:
            cont_list.append(cont[i])
    cont_area = []
    cont_analysis = np.array(cont)[main_hier]
    for i in cont_analysis:
        cont_area.append(cv2.contourArea(i))
    c_keep = np.where((cont_area > np.mean(cont_area)-np.std(cont_area))&(cont_area < np.mean(cont_area)+np.std(cont_area)))[0]
    cont_area = np.array(cont_area)[c_keep]
    cont_analysis = cont_analysis[c_keep]
    c_keep = np.where((cont_area > np.mean(cont_area)+np.std(cont_area)))[0]
    cont_area = np.array(cont_area)[c_keep]
    cont_analysis = cont_analysis[c_keep]
    return cont_area,cont_analysis

def width_angle(pool,cont_analysis):
	global t_dists_list
	t_dists = np.array(pool.map(partial(mp_t_dists,cont_a=cont_analysis),range(0,len(cont_analysis))))
	t_dists_list = t_dists.copy()
	t_dists = t_dists[np.where(t_dists>-50)]
	t_hist, t_bins = np.histogram(t_dists,45,range=[-45,45])
	t_bins = t_bins[1:]-(t_bins[1]-t_bins[0])/2
	arg_max = np.argmax(t_hist)
	fit_hist = t_hist[max(arg_max-7,0):arg_max+7]
	fit_bins = t_bins[max(arg_max-7,0):arg_max+7]
	try:
		_,t_mean,_ = minimize(lambda coefs: np.sum((fit_hist-coefs[0]*np.exp(-0.5*((fit_bins-coefs[1])/coefs[2])**2))**2),x0=[max(fit_hist),fit_bins[np.argmax(fit_hist)],1],bounds=([0,max(t_hist)*2],[-45,45],[0.001,20])).x
		return t_mean
	except:
		return t_bins[np.argmax(t_hist)]

def width_rotate(cont_analysis,ang):
    c_a = []
    for i in cont_analysis:
        c_a.append(rotate_cont(i,ang).round().astype(int))
    return np.array(c_a,dtype=np.object)

def width_template_detect(pool,cont_analysis):
    data = pool.map(partial(mp_template_2,cont_list=cont_analysis),range(0,len(cont_analysis)))
    widths = []
    heights = []
    for i in data:
        if (i[0] == -1):
            continue
        widths.append(i[0])
        heights.append(i[1])
    if len(widths) == 0 or len(heights) == 0:
        return 0, 0
    return np.mean(np.array(widths)[ np.where((widths < np.mean(widths)+np.std(widths))&(widths > np.mean(widths)-np.std(widths)))[0] ]).round().astype(np.uint8),np.mean(np.array(heights)[ np.where((heights < np.mean(heights)+np.std(heights))&(heights > np.mean(heights)-np.std(heights)))[0] ]).round().astype(np.uint8)

def width_template_create(w_h,w_v,ang):
    space = int((w_h+w_v)*0.45)*2
    x_len = w_h+np.round(space*2).astype(int)
    y_len = (w_v+np.round(space*2)).astype(int)
    temp = np.zeros([y_len,x_len])
    temp[space:w_v+space,space:w_h+space]=np.ones([w_v,w_h])
    temp[:space,space:w_h+space]=np.ones([space,w_h])
    temp[w_v+space:,space:w_h+space]=np.ones([space,w_h])
    temp[space:w_v+space,:space]=np.ones([w_v,space])
    temp[space:w_v+space,w_h+space:]=np.ones([w_v,space])

    temp = imutils.rotate(temp,ang)

    space = int((w_h+w_v)*0.45)
    x_len = w_h+np.round(space*2).astype(int)
    y_len = (w_v+np.round(space*2)).astype(int)

    temp = temp[(temp.shape[0]-y_len)//2:(temp.shape[0]-y_len)//2+y_len,(temp.shape[1]-x_len)//2:(temp.shape[1]-x_len)//2+x_len]
    temp = (gaussian_filter(temp,sigma=3)>0.8).astype(np.uint8)
    return temp

def width_perform_ncc(binary,temp):
    temp_binary = np.zeros([binary.shape[0]+temp.shape[0]-1,binary.shape[1]+temp.shape[1]-1],dtype=np.uint8)
    temp_binary[temp.shape[0]//2:binary.shape[0]+temp.shape[0]//2,temp.shape[1]//2:binary.shape[1]+temp.shape[1]//2] = cv2.threshold(binary,254,1,cv2.THRESH_BINARY_INV)[1]
    return cv2.matchTemplate(temp_binary,temp.astype(np.uint8),cv2.TM_CCORR_NORMED)

def moments_retr(c):
	M = cv2.moments(c)
	return [M["m00"],M["m10"],M["m01"]]

def width_find_centres(image,cor,threshold):
	cor_binary = (cor>=threshold).astype(np.uint8)
	p_cont,_ = cv2.findContours(cor_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	p_centres = []
	moment_list = np.array(list(map(moments_retr,p_cont)))
	moment_list = moment_list[moment_list[:,0] > 0]
	cX = moment_list[:,1]/moment_list[:,0]
	cY = moment_list[:,2]/moment_list[:,0]
	p_centres = np.array([cX,cY]).astype(int).T
	return p_centres

def width_auto_thres(cor,image,temp):
    low_bound = minimize_scalar(max_std,args=(cor,image,temp), bounds=[0.7,0.9],method="bounded")
    res = minimize_scalar(min_std,args=(cor,image,temp), bounds=[low_bound.x,np.amax(cor)],method="bounded")
    hist, bins = np.histogram(cor.reshape(1,-1),(100/(res.x-low_bound.x)).astype(int))
    bins = (np.delete(bins,len(bins)-1)+np.delete(bins,0))/2
    return find_half_area(hist,bins,low_bound.x,res.x)

def width_contrast(image,temp,p_centres):
    bright = []
    dark = []
    start_x = temp.shape[1]//2
    end_x = temp.shape[1]//2 + temp.shape[1]%2
    start_y = temp.shape[0]//2
    end_y = temp.shape[0]//2 + temp.shape[0]%2
    inv_temp = 1-temp
    for p in p_centres:
        x0 = p[0]-start_x
        x1 = p[0]+end_x
        y0 = p[1]-start_y
        y1 = p[1]+end_y
        if (y0 < 0) or (y1>image.shape[0]) or (x0 < 0) or (x1>image.shape[1]): 
            bright.append(-1)
            dark.append(-1)
            continue
        im = image[y0:y1,x0:x1]
        bright.append(np.mean(im*temp))
        dark.append(np.mean(im*inv_temp))
    temp_size = temp.shape[0]*temp.shape[1]
    dark_size = len(np.where(inv_temp==1)[0])
    bright_size = len(np.where(temp==1)[0])
    d_bright=(np.array(bright)/bright_size-np.array(dark)/dark_size)*temp_size
    outliers = np.where((d_bright<np.mean(d_bright)-2*np.std(d_bright))&(d_bright>0))[0]
    outliers = outliers[np.where(np.array(bright)[outliers] >= 0)[0]]
    return np.delete(p_centres,outliers,0)

def width_distance(cor,p_centres,auto_thres=True,thres=0):
	distances = cdist(p_centres,p_centres)
	pair = np.argsort(distances)
	nearest = distances[list(range(0,len(distances))),pair[:,1]]
	pair = pair[:,1]
	del distances
	if (auto_thres):
		sorted_near = np.sort(nearest)
		gau_nearest = gaussian_filter1d(sorted_near,sigma=1,order=1)
		near_lim_high = np.where(sorted_near > np.mean(nearest)-np.std(nearest))[0]
		near_lim_high = near_lim_high[0] if len(near_lim_high) > 0 else -1
		near_lim_low = np.where(sorted_near > np.mean(nearest)-2*np.std(nearest))[0]
		near_lim_low = near_lim_low[0] if len(near_lim_low) > 0 else 0
		near_wh = np.where(gau_nearest[:near_lim_high] > np.mean(gau_nearest[:near_lim_high])+2*np.std(gau_nearest[:near_lim_high]))[0]
		if len(near_wh):
			near_threshold = sorted_near[near_wh[-1]]
			#near_threshold = sorted_near[np.where(gau_nearest == np.max(gau_nearest[near_lim_low:near_lim_high]))[0]]
			outliers = np.where(nearest <= near_threshold)[0]
			outliers = outliers[np.argsort(nearest[outliers])]
			near_outliers = nearest[outliers]
		else:
			outliers = np.where(nearest <= near_lim_low)[0]
			outliers = outliers[np.argsort(nearest[outliers])]
			near_outliers = nearest[outliers]
	else:
		outliers = np.where(nearest <= thres*np.std(nearest))[0]
		outliers = outliers[np.argsort(nearest[outliers])]
		near_outliers = nearest[outliers]

	rem_centres = []

	for h in np.array([list(j)[0] for i, j in groupby(near_outliers)]):
		loc_outlier = outliers[np.where(near_outliers == h)[0][0]]
		cor_1 = cor[p_centres[loc_outlier][1],p_centres[loc_outlier][0]]
		cor_2 = cor[p_centres[pair[loc_outlier]][1],p_centres[pair[loc_outlier]][0]]
		if (cor_1 > cor_2):
			rem_centres.append(pair[loc_outlier])
		else:
			rem_centres.append(loc_outlier)

	return np.delete(p_centres,rem_centres,0)

def gaussian_minimum(coefs,x,y):
	ans = coefs[-1]
	for i in range(len(coefs)//3):
		ans += abs(coefs[i*3])*np.exp(-0.5*((x-coefs[i*3+1])/coefs[i*3+2])**2)
	a = y-ans
	return np.sum((a)**2)

def gaussian(coefs,x):
	ans = coefs[-1]
	for i in range(len(coefs)//3):
		ans += abs(coefs[i*3])*np.exp(-0.5*((x-coefs[i*3+1])/coefs[i*3+2])**2)
	return ans

def deconvolute_max_signal_2(x,y):
    data_y = y
    data_x = x
    data_const = 0
    #lims = find_limits(data_x,data_y-data_const,np.argmax(data_y))
    lims = [0,len(data_x)-1]
    if lims[0] > 0:
        lims[0] -= 1
    if lims[1] < len(data_y)-1:
        lims[1] += 1
    min_fun = partial(gaussian_minimum,x=data_x,y=data_y)
    b = [ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
    f = []
    lin_a,lin_mean, lin_sig = np.meshgrid(np.linspace(max(abs(data_y))/10,max(abs(data_y))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],50),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50))
    lin_a = np.concatenate(np.concatenate(lin_a))
    lin_mean = np.concatenate(np.concatenate(lin_mean))
    lin_sig = np.concatenate(np.concatenate(lin_sig))
    for i in range(len(lin_mean)):
        p0 = [lin_a[i], lin_mean[i], lin_sig[i], data_const]
        first_fit = min_fun(p0)
        f += [first_fit]
    f = np.array(f)
    p0 = [lin_a[np.argmin(f)], lin_mean[np.argmin(f)], lin_sig[np.argmin(f)],data_const]
    first_fit = minimize(min_fun,p0,bounds=b)
    resultant = data_y[lims[0]:lims[1]]-gaussian(first_fit.x,data_x[lims[0]:lims[1]])
    b1 = list(b[:-1]) +[ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
    f = []
    lin_a,lin_mean, lin_sig = np.meshgrid(np.linspace(max(abs(resultant))/10,max(abs(resultant))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],50),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50))
    lin_a = np.concatenate(np.concatenate(lin_a))
    lin_mean = np.concatenate(np.concatenate(lin_mean))
    lin_sig = np.concatenate(np.concatenate(lin_sig))
    for i in range(len(lin_mean)):
        p1 = list(first_fit.x[:-1])+[lin_a[i], lin_mean[i], lin_sig[i]]+[data_const]
        second_fit = min_fun(p1)
        f += [second_fit]
    f = np.array(f)
    p1 = list(first_fit.x[:-1])+[lin_a[np.argmin(f)], lin_mean[np.argmin(f)], lin_sig[np.argmin(f)]]+[data_const]
    second_fit = minimize(min_fun,p1,bounds=b1)
    resultant = data_y[lims[0]:lims[1]]-gaussian(second_fit.x,data_x[lims[0]:lims[1]])
    f = []
    b2 = list(b1[:-1]) +[ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
    lin_a,lin_mean, lin_sig = np.meshgrid(np.linspace(max(abs(resultant))/10,max(abs(resultant))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],50),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50))
    lin_a = np.concatenate(np.concatenate(lin_a))
    lin_mean = np.concatenate(np.concatenate(lin_mean))
    lin_sig = np.concatenate(np.concatenate(lin_sig))
    for i in range(len(lin_mean)):
        p2 = list(second_fit.x[:-1])+[lin_a[i], lin_mean[i], lin_sig[i]]+[data_const]
        third_fit = min_fun(p2)
        f += [third_fit]
    f = np.array(f)
    p2 = list(second_fit.x[:-1])+[lin_a[np.argmin(f)], lin_mean[np.argmin(f)], lin_sig[np.argmin(f)]]+[data_const]
    third_fit = minimize(min_fun,p2,bounds=b2)
    resultant = data_y[lims[0]:lims[1]]-gaussian(third_fit.x,data_x[lims[0]:lims[1]])
    f = []
    b3 = list(b2[:-1]) +[ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
    lin_a,lin_mean, lin_sig = np.meshgrid(np.linspace(max(abs(resultant))/10,max(abs(resultant))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],50),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50))
    lin_a = np.concatenate(np.concatenate(lin_a))
    lin_mean = np.concatenate(np.concatenate(lin_mean))
    lin_sig = np.concatenate(np.concatenate(lin_sig))
    for i in range(len(lin_mean)):
        p3 = list(third_fit.x[:-1])+[lin_a[i], lin_mean[i], lin_sig[i]]+[data_const]
        fourth_fit = min_fun(p3)
        f += [fourth_fit]
    f = np.array(f)
    p3 = list(third_fit.x[:-1])+[lin_a[np.argmin(f)], lin_mean[np.argmin(f)], lin_sig[np.argmin(f)]]+[data_const]
    fourth_fit = minimize(min_fun,p3,bounds=b3)
    return fourth_fit.x

def gaussian_array(x,a,mean,sigma):
	return a*np.exp(-0.5*((x-mean)/sigma)**2)

def deconvolute_max_signal(x,y):
	data_y = y
	data_x = x
	data_const = 0
	lims = [0,len(data_y)-1]
	if lims[0] > 0:
		lims[0] -= 1
	if lims[1] < len(data_y)-1:
		lims[1] += 1
	min_fun = partial(gaussian_minimum,x=data_x,y=data_y)
	b = [ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
	#First Fit
	lin_a, lin_mean, lin_sig, lin_x = np.meshgrid(np.linspace(max(abs(data_y))/10,max(abs(data_y))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],50),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50),data_x[lims[0]:lims[1]])
	data_x_len = len(data_x[lims[0]:lims[1]])
	f = gaussian_array(lin_x,lin_a,lin_mean,lin_sig)
	f = np.sum((data_y[lims[0]:lims[1]]-f.reshape(-1,data_x_len)-data_const)**2,axis=1)
	lin_a = lin_a.reshape(-1,data_x_len)[:,0]
	lin_mean = lin_mean.reshape(-1,data_x_len)[:,0]
	lin_sig = lin_sig.reshape(-1,data_x_len)[:,0]
	min_f_arg = np.argmin(f)
	p0 = [ lin_a[ min_f_arg ] , lin_mean[ min_f_arg ] , lin_sig[ min_f_arg ] ,data_const]
	first_fit = minimize(min_fun,p0,bounds=b)
	resultant = data_y[lims[0]:lims[1]]-gaussian(first_fit.x,data_x[lims[0]:lims[1]])
	#Second Fit
	b1 = list(b[:-1]) +[ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
	lin_a, lin_mean, lin_sig, lin_x = np.meshgrid(np.linspace(max(abs(resultant))/100,max(abs(resultant))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],20),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50),data_x[lims[0]:lims[1]])
	f = gaussian_array(lin_x,lin_a,lin_mean,lin_sig)
	f = np.sum((resultant-f.reshape(-1,data_x_len))**2,axis=1)
	lin_a = lin_a.reshape(-1,data_x_len)[:,0]
	lin_mean = lin_mean.reshape(-1,data_x_len)[:,0]
	lin_sig = lin_sig.reshape(-1,data_x_len)[:,0]
	min_f_arg = np.argmin(f)
	p1 = list(first_fit.x[:-1])+[ lin_a[ min_f_arg ] , lin_mean[ min_f_arg ] , lin_sig[ min_f_arg ] ]+[first_fit.x[-1]]
	second_fit = minimize(min_fun,p1,bounds=b1)
	resultant = data_y[lims[0]:lims[1]]-gaussian(second_fit.x,data_x[lims[0]:lims[1]])
	#Third Fit
	b2 = list(b1[:-1]) +[ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
	lin_a, lin_mean, lin_sig, lin_x = np.meshgrid(np.linspace(max(abs(resultant))/100,max(abs(resultant))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],20),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50),data_x[lims[0]:lims[1]])
	f = gaussian_array(lin_x,lin_a,lin_mean,lin_sig)
	f = np.sum((resultant-f.reshape(-1,data_x_len))**2,axis=1)
	lin_a = lin_a.reshape(-1,data_x_len)[:,0]
	lin_mean = lin_mean.reshape(-1,data_x_len)[:,0]
	lin_sig = lin_sig.reshape(-1,data_x_len)[:,0]
	min_f_arg = np.argmin(f)
	p2 = list(second_fit.x[:-1])+[ lin_a[ min_f_arg ] , lin_mean[ min_f_arg ] , lin_sig[ min_f_arg ] ]+[second_fit.x[-1]]
	third_fit = minimize(min_fun,p2,bounds=b2)
	resultant = data_y[lims[0]:lims[1]]-gaussian(third_fit.x,data_x[lims[0]:lims[1]])
	#Fourth Fit
	b3 = list(b2[:-1]) +[ [0,max(data_y)*2], [data_x[lims[0]],data_x[lims[1]]] , [-(data_x[lims[1]]-data_x[lims[0]]) , data_x[lims[1]]-data_x[lims[0]]] , [0,0] ]
	lin_a, lin_mean, lin_sig, lin_x = np.meshgrid(np.linspace(max(abs(resultant))/100,max(abs(resultant))*2,10),np.linspace(data_x[lims[0]],data_x[lims[1]],20),np.linspace((data_x[lims[1]]-data_x[lims[0]])/1000,data_x[lims[1]]-data_x[lims[0]],50),data_x[lims[0]:lims[1]])
	f = gaussian_array(lin_x,lin_a,lin_mean,lin_sig)
	f = np.sum((resultant-f.reshape(-1,data_x_len))**2,axis=1)
	lin_a = lin_a.reshape(-1,data_x_len)[:,0]
	lin_mean = lin_mean.reshape(-1,data_x_len)[:,0]
	lin_sig = lin_sig.reshape(-1,data_x_len)[:,0]
	min_f_arg = np.argmin(f)
	p3 = list(third_fit.x[:-1])+[ lin_a[ min_f_arg ] , lin_mean[ min_f_arg ] , lin_sig[ min_f_arg ] ]+[third_fit.x[-1]]
	fourth_fit = minimize(min_fun,p3,bounds=b3)
	return fourth_fit.x

def detect_image_thres(image):
	hist = cv2.calcHist([image],[0],None,[256],[0,256]).T[0].astype(int)
	hist[-2] += hist[-1]
	hist = hist[:-1]
	bins = np.arange(256)
	bins = bins[1:]-0.5
	hist = np.delete(hist,np.arange(110,128))
	bins = np.delete(bins,np.arange(110,128))
	hist = np.delete(hist,np.arange(0,2))
	bins = np.delete(bins,np.arange(0,2))
	hist = np.delete(hist,np.arange(len(bins)-2,len(bins)))
	bins = np.delete(bins,np.arange(len(bins)-2,len(bins)))
	deconv = deconvolute_max_signal(bins,hist)
	a_list = np.array([deconv[0],deconv[3],deconv[6],deconv[9]])
	mean_list = np.array([deconv[1],deconv[4],deconv[7],deconv[10]])
	std_list = np.array([deconv[2],deconv[5],deconv[8],deconv[11]])
	area_list = np.abs(a_list*std_list)
	max_area = np.argmax(area_list)*3
	l_bg = np.where(mean_list-1.5*np.abs(std_list) > deconv[max_area+1])[0]
	if len(l_bg) > 0:
		high_mean = l_bg[np.argmin(mean_list[l_bg])]
		return deconv[high_mean*3+1]-1.5*abs(deconv[high_mean*3+2])
	else:
		return deconv[max_area+1]

def width_binary(image,thres,sig=2.5):
	binary = cv2.threshold(image,thres,255,cv2.THRESH_BINARY_INV)[1]
	binary = cv2.GaussianBlur(binary,(9,9),sig)
	return (binary > np.mean(binary)).astype(np.uint8)*255

def individual_plot(coefs,x,y):
	plt.plot(x,y)
	ans = 0
	for i in range(len(coefs)//3):
		t_ans = abs(coefs[i*3])*np.exp(-0.5*((x-coefs[i*3+1])/coefs[i*3+2])**2)
		plt.plot(x,t_ans)
		ans += t_ans
	plt.plot(x,ans)
	return plt.show()

def width_auto_process(pool,image):
	time_start = time.time()
	#Perform a bandpass filter!
	fft_time = time.time()
	print("Performing FFT.")
	#Performs FFT on the image using the GPU. If unable, uses CPU instead.
	image = perform_fft(image)
	#image = perform_fft_cpu(image)
	print(" Time taken for FFT",time.time()-fft_time,"seconds.")
	print("Creating binary image.")
	binary = width_binary(image,detect_image_thres(image))
	#Detect closed shapes
	cont_area,cont_analysis = width_contours(binary)

	print("Dendrite rotation angle detection.")
	ang = width_angle(pool,cont_analysis)
	print("Detected angle: "+str(-ang))
	if np.isnan(ang):
		ang = 0
	#Rotate contours
	print("Rotating contours.")
	cont_analysis=width_rotate(cont_analysis,ang)
	print("Performing template creation.")
	w_h,w_v = width_template_detect(pool,cont_analysis)
	#Create template
	global temp, cor
	temp = width_template_create(w_h,w_v,ang)

	cor = width_perform_ncc(binary,temp)
	print("Performing threshold detections.")
	global thres
	thres = width_auto_thres(cor,image,temp)
	p_centres = width_find_centres(image,cor,thres)
	#Remove outliers detected by contrast differences
	p_centres = width_contrast(image,temp,p_centres)
	#Remove outliers detected by closest distance
	p_centres = width_distance(cor,p_centres)

	time_taken = time.time()-time_start
	print("Time to process image: "+str(time_taken))
	return image,np.array(p_centres)
