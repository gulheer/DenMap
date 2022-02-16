import denmap_src.width_gui as wg
import numpy as np
import multiprocessing as mp

def progress_fft(image, r_in, r_out, prog_queue, res_queue, shared_memory, factors = False, CPU = False):
	CPU_success = True
	GPU_success = True
	tot_num = 100/4
	prog_queue.put([0,"Calculating Gaussian matrix for bandpass filtering"])
	if type(factors) == bool:
		factors = wg.gaussian_matrix_fft(image,r_in,r_out)
	if not CPU and not shared_memory:
		prog_queue.put([tot_num,"Attempting GPU FFT"])
		image_dev, thr, gpu_fft, gpu_fftshift = wg.attempt_gpu_fft(image)
	else:
		image_dev = False
	if type(image_dev) == bool or CPU:
		try:
			tot_num = 100/5
			if shared_memory:
				prog_queue.put([2*tot_num,"Performing CPU FFT"])
			else:
				prog_queue.put([2*tot_num,"GPU FFT Failed - Using CPU FFT"])
			image = wg.cpu_fft(image)
			prog_queue.put([3*tot_num,"Multiplying FFT by Gaussian matrix"])
			image = wg.cpu_multiply_factors(image,factors)
			prog_queue.put([4*tot_num,"Peforming inverse FFT"])
			del factors
			image = wg.cpu_ifft(image)
		except:
			CPU_success = False
	else:
		try:
			prog_queue.put([2*tot_num,"Multiplying FFT by Gaussian matrix"])
			wg.gpu_multiply_factors(image_dev,thr,factors)
			del factors
			prog_queue.put([3*tot_num,"Peforming inverse FFT"])
			image = wg.inverse_gpu_fft(image_dev, thr, gpu_fft, gpu_fftshift)
			#thr.synchronize()
			thr.release()
		except:
			GPU_success = False

	if CPU_success and GPU_success:
		prog_queue.put([100,"Finishing"])
		res_queue.put(wg.perform_image_contrast(image).astype(np.uint8))
	elif not GPU_success and CPU_success:
		try:
			progress_fft(image, r_in, r_out, prog_queue, res_queue, shared_memory, factors, CPU = True)
		except:
			progress_fft(image, r_in, r_out, prog_queue, res_queue, shared_memory, factors=False, CPU = True)
	else:
		prog_queue.put([0,""])
		res_queue.put(-1)

def auto_process_thread(new_image, original_image, circle_image, binary_image, res_queue, prog_queue, shared_memory):
    tot_num_prog = 9
    cpu_count = mp.cpu_count()
    pool = mp.Pool(processes=cpu_count)
    if type(new_image)==bool and type(binary_image) == bool:
        prog_queue.put([0,"Performing FFT"])
    #Performs FFT on the image using the GPU. If unable, uses CPU instead.
        image = wg.perform_fft(original_image,shared_memory)
        if type(image) == bool:
            res_queue.put([-2])
            return
    elif not type(new_image)==bool:
        image = new_image
    else:
        image = original_image
    
    if type(binary_image) == bool:
        prog_queue.put([100/tot_num_prog,"Creating binary image"])
        binary = wg.width_binary(image,wg.detect_image_thres(image))
    else:
        binary = binary_image

    prog_queue.put([2*100/tot_num_prog,"Performing contour analysis"])
    #Detect closed shapes
    cont_area,cont_analysis = wg.width_contours(binary)
    prog_queue.put([3*100/tot_num_prog,"Performing dendrite rotation angle detection"])
    ang = wg.width_angle(pool,cont_analysis)
    if np.isnan(ang):
        ang = 0
    #Rotate contours
    prog_queue.put([4*100/tot_num_prog,"Rotating contours"])

    cont_analysis=wg.width_rotate(cont_analysis,ang)
    prog_queue.put([5*100/tot_num_prog,"Generating template"])
    w_h,w_v = wg.width_template_detect(pool,cont_analysis)
    #Create template
    if w_h == 0 or w_v == 0:
        res_queue.put([-1])
        return
    temp = wg.width_template_create(w_h,w_v,ang)

    prog_queue.put([6*100/tot_num_prog,"Performing NCC"])
    cor = wg.width_perform_ncc(binary,temp)

    prog_queue.put([7*100/tot_num_prog,"Detecting thresholds"])
    thres = wg.width_auto_thres(cor,image,temp)
    prog_queue.put([8*100/tot_num_prog,"Finding centres"])
    p_centres = wg.width_find_centres(image,cor,thres)
    #Remove outliers detected by contrast differences
    p_centres = wg.width_contrast(image,temp,p_centres)
    #Remove outliers detected by closest distance
    p_centres = wg.width_distance(cor,p_centres)
    prog_queue.put([100,"Finished"])
    res_queue.put([image,cor,thres,temp,p_centres])
    pool.close()