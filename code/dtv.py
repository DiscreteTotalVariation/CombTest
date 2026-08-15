# -*- coding: utf-8 -*-

EXACT_DISTRIBUTIONS_DIR="/mnt/data/tvor/exact_distributions"
MEAN_AND_VAR_FOR_EXACT_DISTRIBUTIONS_DIR="/mnt/data/tvor/mean_and_var_for_exact_distributions"
ESTIMATED_MEAN_AND_VAR_FOR_SIMULATED_DISTRIBUTIONS_DIR="/mnt/data/tvor/estimated_mean_and_var_for_simulated_distributions"

OBTAINED_P_AND_DTV_VALUES_DIR="/mnt/data/tvor/obtained_p_and_dtv_values"

HISTOGRAM_CONFIGURATION_PATTERN="N_%d_n_%d.txt"

# such a big value is effectively an overkill designed to eliminate any potential precision problems
MEAN_AND_VAR_WORKING_DECIMAL_PRECISION=2**16
MEAN_AND_VAR_OUTPUT_DECIMAL_PRECISION=8

CONTINUITY_CORRECTION_FOR_BINOMIAL_TO_NORMAL=0.5
CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA=0.5

def load_lines(path, encoding=None):
	lines=[]
	if (encoding is None):
		with open(path, "r") as f:
			lines=[l.strip("\r\n") for l in f.readlines()]
	else:
		with open(path, "r", encoding=encoding) as f:
			lines=[l.strip("\r\n") for l in f.readlines()]
	return lines

def get_distribution_from_dtv_counts(counts, N):
	# count the occurrence of every possible value of the discrete total variation for the whole histogram
	dtv_counts=dict()
	for last in range(0, N+1):
		for dtv in range(0, 2*N+1):
			count=counts[N][last][dtv]
			dtv_counts[dtv]=dtv_counts.get(dtv, 0)+count

	# prepare the final result
	distribution=[]
	for dtv in range(max(dtv_counts.keys())+1):
		distribution.append((dtv, dtv_counts.get(dtv, 0)))

	# return the final result
	return distribution

def calculate_exact_dtv_distribution(N, n, show_status=False, cache_dir=None):
	if (n==0):
		# if there are no bins, return an empty distribution
		return []

	# if caching the results is expected, then the case for n=1 is the simplest
	if (cache_dir is not None):
		from os import makedirs
		makedirs(cache_dir, exist_ok=True)
		from os.path import join
		with open(join(cache_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, 1)), "w", encoding="utf8") as fo:
			fo.write("0 1\n")
			for i in range(1, 2*N+1):
				fo.write("%d 0\n"%(i))

	if (n==1):
		# if there is only a single bin, than the only possible value for the DTV is 0, i.e., all other values cannot appear by definition
		return [(0, 1)]+[(x, 0) for x in range(2*N+1)]
	
	# the structure of the previous and the current counts is the same and the value counts[sum][last_bin_value][dtv] tells how often the discrete total variation equals dtv if the size of the subsample falling in the currently examined part of the histogram (the histogram is being examined bin by bin in a for loop) is sum and the count in the last histogram is last_bin_value
	previous_counts=[]
	current_counts=[]
	# the usual Python lists are used because the results will be stored in the unbouned plain Python int type
	for j in range(N+1):
		previous_counts2=[]
		current_counts2=[]
		for k in range(N+1):
			previous_counts2.append([0]*(2*N+1+1))
			current_counts2.append([0]*(2*N+1+1))
		previous_counts.append(previous_counts2)
		current_counts.append(current_counts2)

	from math import comb

	# initializing the initial counts for the first bin
	for i in range(N+1):
		current_counts[i][i][0]=comb(N, i)

	# this will contain the final result as well as the intermediate results if caching is required
	distribution=None

	# since the results for the first bin have already been calculated above, the loop starts from the second bin, whose index in zero-based indexing is 1
	taker=list(range(1, n))
	if (show_status is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	# the main loop that processes every histogram bin by bin based on the result obtained for the previous bin
	for bin_idx in taker:
		# to save memory, only two counts variables are used and here their role is swapped
		previous_counts, current_counts=current_counts, previous_counts
		# clearing the current counts
		for i in range(len(current_counts)):
			for j in range(len(current_counts[i])):
				for k in range(len(current_counts[i][j])):
					current_counts[i][j][k]=0
		# check every possible sum, i.e., size of the subsample that could have fallen into the previously examined part of the histogram
		for previous_sum in range(0, N+1):
			# the size of the subsample that falls into the remaining n-bin_idx-1 bins
			remaining=N-previous_sum
			# check every possible value of the last bin with respect to the currently observed subsample size
			for previous_last in range(0, previous_sum+1):
				# check every possible DTV value with respect to the currently observed subsample size
				for previous_dtv in range(0, 2*previous_sum+1+1):
					# take the result for the currently observed previous state
					previous_count=previous_counts[previous_sum][previous_last][previous_dtv]
					if (previous_count>0):
						# check every possible transition from the currently observed previous state to a currenty observed current state
						for new_last in range(0, remaining+1):
							new_sum=previous_sum+new_last
							new_dtv=previous_dtv+abs(new_last-previous_last)
							current_counts[new_sum][new_last][new_dtv]+=previous_count*comb(remaining, new_last)
							

		# if caching is required, then the current result should be stored
		if (cache_dir is not None):
			distribution=get_distribution_from_dtv_counts(counts=current_counts, N=N)
			with open(join(cache_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, bin_idx+1)), "w", encoding="utf8") as fo:
				for dtv, count in distribution:
					fo.write("%d %d\n"%(dtv, count))

	if (distribution is None):
		distribution=get_distribution_from_dtv_counts(counts=current_counts, N=N)
	
	# return the calculated result
	return distribution

def calculate_exact_dtv_distribution2(N, n, show_status=False, cache_dir=None):
	if (n==0):
		# if there are no bins, return an empty distribution
		return []

	# if caching the results is expected, then the case for n=1 is the simplest
	if (cache_dir is not None):
		from os import makedirs
		makedirs(cache_dir, exist_ok=True)
		from os.path import join
		with open(join(cache_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, 1)), "w", encoding="utf8") as fo:
			fo.write("0 1\n")
			for i in range(1, 2*N+1):
				fo.write("%d 0\n"%(i))

	if (n==1):
		# if there is only a single bin, than the only possible value for the DTV is 0, i.e., all other values cannot appear by definition
		return [(0, 1)]+[(x, 0) for x in range(2*N+1)]
	
	# the structure of the previous and the current counts is the same and the value counts[sum][last_bin_value][dtv] tells how often the discrete total variation equals dtv if the size of the subsample falling in the currently examined part of the histogram (the histogram is being examined bin by bin in a for loop) is sum and the count in the last histogram is last_bin_value
	previous_counts=[]
	current_counts=[]
	# the usual Python lists are used because the results will be stored in the unbouned plain Pytonn int type
	for j in range(N+1):
		previous_counts2=[]
		current_counts2=[]
		for k in range(N+1):
			previous_counts2.append([0]*(2*N+1+1))
			current_counts2.append([0]*(2*N+1+1))
		previous_counts.append(previous_counts2)
		current_counts.append(current_counts2)

	from math import comb

	# initializing the initial counts for the first bin
	for i in range(N+1):
		current_counts[i][i][0]=comb(N, i)

	# this will contain the final result as well as the intermediate results if caching is required
	distribution=None

	# since the results for the first bin have already been calculated above, the loop starts from the second bin, whose index in zero-based indexing is 1
	taker=list(range(1, n))
	if (show_status is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	# the main loop that processes every histogram bin by bin based on the result obtained for the previous bin
	for bin_idx in taker:
		# to save memory, only two counts variables are used and here their role is swapped
		previous_counts, current_counts=current_counts, previous_counts
		# clearing the current counts
		for i in range(len(current_counts)):
			for j in range(len(current_counts[i])):
				for k in range(len(current_counts[i][j])):
					current_counts[i][j][k]=0
		for current_sum in range(0, N+1):
			for current_last in range(0, current_sum+1):
				current_combinations=comb(N-current_sum+current_last, current_last)
				for current_dtv in range(0, 2*current_sum+1+1):
					for previous_last in range(0, current_sum-current_last+1):
						if (previous_counts[current_sum-current_last][previous_last][current_dtv-abs(current_last-previous_last)]>0):
							current_counts[current_sum][current_last][current_dtv]+=previous_counts[current_sum-current_last][previous_last][current_dtv-abs(current_last-previous_last)]*current_combinations

		# if caching is required, then the current result should be stored
		if (cache_dir is not None):
			distribution=get_distribution_from_dtv_counts(counts=current_counts, N=N)
			with open(join(cache_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, bin_idx+1)), "w", encoding="utf8") as fo:
				for dtv, count in distribution:
					fo.write("%d %d\n"%(dtv, count))

	if (distribution is None):
		distribution=get_distribution_from_dtv_counts(counts=current_counts, N=N)
	
	# return the calculated result
	return distribution

def load_exact_dtv_distribution(N=None, n=None, input_dir=EXACT_DISTRIBUTIONS_DIR, exact_path=None):
	from os.path import isfile, join

	if (exact_path is not None):
		input_path=exact_path
	else:
		input_path=join(input_dir, "N_%d_n_%d.txt"%(N, n))

	# if the file with the result does not exist, return None
	if (isfile(input_path) is False):
		return None

	distribution=[]
	for l in load_lines(input_path):
		a, b=list(map(int, l.split()))
		distribution.append((a, b))
	return distribution

def calcualte_mean_and_var_for_exact_distribution(distribution, working_precision=MEAN_AND_VAR_WORKING_DECIMAL_PRECISION):
	from decimal import Decimal, getcontext

	getcontext().prec=working_precision

	count=0
	summed=0
	# calculate the components required for calculating the mean value
	for dtv, dtv_count in distribution:
		count+=dtv_count
		summed+=dtv*dtv_count
	# calculate the mean value
	mean=Decimal(summed)/Decimal(count)
	# instead of using the mean value, the sum will be used instead during the calculation of the variance to avoid certain precision problems
	var=0
	for dtv, dtv_count in distribution:
		var+=((count*dtv-summed)**2)*dtv_count
	var=Decimal(var)/Decimal(count**3)

	#return the calculated results
	return mean, var

def load_mean_and_var_for_exact_distribution(N, n):
	from os.path import isfile, join
	input_path=join(MEAN_AND_VAR_FOR_EXACT_DISTRIBUTIONS_DIR, HISTOGRAM_CONFIGURATION_PATTERN%(N, n))
	if (isfile(input_path) is False):
		return None, None
	with open(input_path, "r", encoding="utf8") as f:
		mean, var=list(map(float, f.read().strip("\r\n").split()))
	return mean, var

def load_estimated_mean_and_var_for_simulated_distribution(N, n, simulations_count):
	from os.path import isfile, join
	input_path=join(ESTIMATED_MEAN_AND_VAR_FOR_SIMULATED_DISTRIBUTIONS_DIR, str(simulations_count), HISTOGRAM_CONFIGURATION_PATTERN%(N, n))
	if (isfile(input_path) is False):
		return None, None
	with open(input_path, "r", encoding="utf8") as f:
		mean, var=list(map(float, f.read().strip("\r\n").split()))
	return mean, var

def get_random_sample_histogram(ps, size):
	n=len(ps)
	import numpy as np
	return np.histogram(np.random.choice(list(range(n)), size, p=np.array(ps)/np.sum(ps)), bins=np.array(range(n+1))-0.5)[0]

def get_dtv(d):
	import numpy as np
	return np.sum(np.abs(np.array(d[:-1])-np.array(d[1:])))

def dtv_on_random_sample_worker(ps, size, counter, solution_queue):
	
	while(True):
		with counter.get_lock():
			counter.value-=1
			current=counter.value
		bins=get_random_sample_histogram(ps=ps, size=size)
		dtv=get_dtv(bins)
		solution_queue.put(dtv)
		if (current<=0):
			break

def estimate_mean_and_var_using_monte_carlo(N, n, simulations_count, processes_count=None, verbose=False):
	dtvs=[]

	ps=[1]*n
	taker=list(range(simulations_count))
	if (verbose is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	# if this is not a multiprocessing task, then a simple loop is sufficient
	if (processes_count is None):
		for si in taker:
			bins=get_random_sample_histogram(ps=ps, size=N)
			dtvs.append(get_dtv(bins))
	# otherwise, new processes are created to solve the task
	else:
		import multiprocessing
		solution_queue=multiprocessing.Queue()
		counter=multiprocessing.Value("i", simulations_count)
		workers=[]
		for pi in range(processes_count):
			worker=multiprocessing.Process(target=dtv_on_random_sample_worker, args=(ps, N, counter, solution_queue));
			worker.start()
			workers.append(worker)
		for si in taker:
			dtv=solution_queue.get()
			dtvs.append(dtv)

	import numpy as np
	mean=np.mean(dtvs)
	var=np.var(dtvs)

	return mean, var

def test1():
	N=100
	n=5

	# calculate the occurrences of every possible discrete total variation value
	distribution=calculate_exact_dtv_distribution(N=N, n=n)

	# print the obtained results
	for dtv, count in distribution:
		print("%3d: %d"%(dtv, count))

def test2():
	N=50
	n=4

	# load the occurrences of every possible discrete total variation value
	distribution=load_exact_dtv_distribution(N=N, n=n)

	if (distribution is not None):
		# print the loaded results
		for dtv, count in distribution:
			print("%3d: %d"%(dtv, count))

def test3():
	# the values of N that will be used
	Ns=list(range(1, 1000+1))
	# the upper bound for the value of n
	upper_n=10

	#Ns=Ns[0+0::2]

	verbose=True

	if (verbose is True):
		print("Calculating and caching the exact DTV distributions...")
	for ni, N in enumerate(Ns):
		if (verbose is True):
			print("%d / %d   N = %d"%(ni+1, len(Ns), N))
		# the distributions for value of n lower than upper_n will be cached in the process of calculating the result for upper_n
		calculate_exact_dtv_distribution(N=N, n=upper_n, show_status=verbose, cache_dir=EXACT_DISTRIBUTIONS_DIR)

def test4():
	input_dir=EXACT_DISTRIBUTIONS_DIR
	output_dir=MEAN_AND_VAR_FOR_EXACT_DISTRIBUTIONS_DIR
	verbose=True
	overwrite=False

	from os import listdir, makedirs

	makedirs(output_dir, exist_ok=True)

	from decimal import Decimal, getcontext, ROUND_HALF_UP

	final_precision=MEAN_AND_VAR_OUTPUT_DECIMAL_PRECISION
	quantization_step=Decimal("0."+"0"*(final_precision-1)+"1")

	if (verbose is True):
		print("Calculating the mean and the variance for the existing cached distributions...")
	names=sorted(listdir(input_dir))
	taker=names
	if (verbose is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	from os.path import isfile, join
	# go over the existing cached distributions
	for name in taker:
		input_path=join(input_dir, name)
		output_path=join(output_dir, name)
		# skip this case if it is already calculated and if no overwriting is expected
		if (overwrite is False and isfile(output_path) is True):
			continue
		# load the previously cached distribution
		distribution=load_exact_dtv_distribution(exact_path=input_path)
		
		# calculate the mean and the variance
		mean, var=calcualte_mean_and_var_for_exact_distribution(distribution=distribution)
	
		# perform quantization
		mean=mean.quantize(quantization_step, ROUND_HALF_UP)
		var=var.quantize(quantization_step, ROUND_HALF_UP)
		# prepare the writing format
		mean=format(mean, "f")
		var=format(var, "f")
		# write the obtained results
		with open(output_path, "w", encoding="utf8") as fo:
			fo.write(" ".join(map(str, [mean, var]))+"\n")

def test5():
	verbose=True

	import numpy as np
	
	# in the settings here, n is the size of the sampe

	# the calculation of for the settings below takes some time
	if (False):
		ns=list(range(1, 1000+1))
		ps=np.linspace(0.001, 0.999, 999)
		output_path="bin_to_norm_1000x999.npy"
		output_path2="bin_to_norm_ks_1000x999.npy"
	# the calculation for the settings below is much faster
	if (False):
		ns=list(range(1, 200+1))
		ps=np.linspace(0.01, 0.99, 197)
		output_path="bin_to_norm_200x197.npy"
		output_path2="bin_to_norm_ks_200x197.npy"
	if (False):
		ns=list(range(1, 225+1))
		ps=np.linspace(0.01, 0.99, 197)
		output_path="bin_to_norm_225x197.npy"
		output_path2="bin_to_norm_ks_225x197.npy"
	if (False):
		ns=list(range(1, 250+1))
		ps=np.linspace(0.01, 0.99, 197)
		output_path="bin_to_norm_250x197.npy"
		output_path2="bin_to_norm_ks_250x197.npy"
	if (False):
		ns=list(range(1, 300+1))
		ps=np.linspace(0.01, 0.99, 295)
		output_path="bin_to_norm_300x295.npy"
		output_path2="bin_to_norm_ks_300x295.npy"
	if (True):
		ns=list(range(1, 400+1))
		ps=np.linspace(0.01, 0.99, 393)
		output_path="bin_to_norm_300x295.npy"
		output_path2="bin_to_norm_ks_300x295.npy"
	
	from scipy.stats import binom, norm

	results=dict()
	results2=dict()
	taker=ns
	if (verbose is True):
		print("Assessing the accuracy of using the normal distribution to approximate the binomial distribution...")
		from tqdm import tqdm
		taker=tqdm(taker)
	# go over the values of n
	for n in taker:
		# go over the values of p
		for p in ps:
			q=1-p
			
			# prepare the normal distribution used to approximate the binomial distribution for the given parameters
			nd=norm(loc=n*p, scale=np.sqrt(n*p*q))

			# calculate the mean absolute error of CDF for every value from 0 to n
			score=0
			count=0
			ks=None
			for i in range(n+1):
				exact_value=binom.cdf(i, n, p)
				approximated_value=nd.cdf(i+CONTINUITY_CORRECTION_FOR_BINOMIAL_TO_NORMAL)
				current_score=abs(exact_value-approximated_value)
				if ks is None or ks<current_score:
					ks=current_score
				score+=current_score
				count+=1
			score/=count
			results[(n, p)]=score
			results2[(n, p)]=ks
	
	# prepare the results for saving
	data=[]
	for n in ns:
		current=[]
		for p in ps:
			current.append(results[(n, p)])
		data.append(current)
	data=np.array(data)
	data2=[]
	for n in ns:
		current=[]
		for p in ps:
			current.append(results2[(n, p)])
		data2.append(current)
	data2=np.array(data2)
	# save the obtained results
	np.save(output_path, data)
	np.save(output_path2, data2)

def test6():
	import numpy as np

	# in the settings here, n is the size of the sampe

	# the larger plot display, but without the focus on the problematic areas
	if (False):
		xticks=[0, 124, 249, 374, 499, 624, 749, 874, 999]
		xtick_labels=[1, 125, 250, 375, 500, 625, 750, 875, 1000]
		yticks=[0, 124, 249, 374, 499, 624, 749, 874, 998]
		ytick_labels=[0.99, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.01]
		input_path="bin_to_norm_1000x999.npy"
		save_path="bin_to_norm_1000x999.png"
	# the smaller plot display with a better focus on the problematic areas
	if (False):
		xticks=[0, 24, 49, 74, 99, 124, 149, 174, 199]
		xtick_labels=[1, 25, 50, 75, 100, 125, 150, 175, 200]
		yticks=[0, 23, 48, 73, 98, 123, 148, 173, 196]
		ytick_labels=[0.99, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.01]
		input_path="bin_to_norm_200x197.npy"
		save_path="bin_to_norm_200x197.png"
	if (True):
		xticks=[0, 49, 99, 149, 199, 249, 299]
		xtick_labels=[1, 50, 100, 150, 200, 250, 300]
		yticks=[0, 73, 147, 220, 294]
		ytick_labels=[0.99, 0.75, 0.5, 0.25, 0.01]
		input_path="bin_to_norm_300x295.npy"
		save_path="bin_to_norm_300x295.png"

	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30
	
	# the image saving settings
	dpi=300
	format="png"

	data=np.load(input_path)
	# this will put n to the x-axis and p to the y-axis
	data=data.T
	# this will assure that the final image will have a commonly used orientation
	data=data[::-1, :]

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	plt.ticklabel_format(style="plain")
	plt.imshow(data, cmap="turbo", interpolation="nearest")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.xticks(xticks, xtick_labels)
	plt.yticks(yticks, ytick_labels)
	plt.ylabel("$p$", fontsize=label_fontsize)
	plt.colorbar()
	plt.clim(0, 0.05)
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test7():
	verbose=True

	if (False):
		Ns=list(range(2, 200+1))
		ns=list(range(2, 200+1))
		output_path="dtv_to_gamma_199x199.npy"
	if (False):
		Ns=list(range(2, 225+1))
		ns=list(range(2, 225+1))
		output_path="dtv_to_gamma_224x224.npy"
	if (False):
		Ns=list(range(2, 250+1))
		ns=list(range(2, 250+1))
		output_path="dtv_to_gamma_249x249.npy"
	if (True):
		Ns=list(range(2, 300+1))
		ns=list(range(2, 300+1))
		output_path="dtv_to_gamma_299x299.npy"

	from scipy.stats import gamma
	
	results=dict()
	taker=Ns
	if (verbose is True):
		print("Assessing the accuracy of using the gamma distribution to approximate the DTV distribution...")
		from tqdm import tqdm
		taker=tqdm(taker)
	from tqdm import tqdm
	# go over the values of N
	for N in taker:
		# go over the values of n
		for n in ns:
			
			distribution=load_exact_dtv_distribution(N=N, n=n)

			mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)

			gamma_alpha=(mean**2)/var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=var/mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			# this sum will be used to calculate the exact values of CDF
			summed=sum([dtv_count for dtv, dtv_count in distribution])

			# calculate the mean absolute error of CDF for every value from 0 to 2*N
			score=0
			count=0
			cummulative=0
			for i in range(0, 2*N+1):
				cummulative+=distribution[i][1]
				exact_value=cummulative/summed
				approximated_value=g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				score+=abs(approximated_value-exact_value)
				count+=1
			score/=count
			results[(N, n)]=score

	# prepare the results for saving
	import numpy as np
	data=[]
	for N in Ns:
		current=[]
		for n in ns:
			current.append(results[(N, n)])
		data.append(current)
	data=np.array(data)
	# save the obtained results
	np.save(output_path, data)

def test8():
	if (False):
		xticks=[0, 23, 48, 73, 98, 123, 148, 173, 198]
		xtick_labels=[2, 25, 50, 75, 100, 125, 150, 175, 200]
		yticks=[0, 23, 48, 73, 98, 123, 148, 173, 198]
		ytick_labels=[200, 175, 150, 125, 100, 75, 50, 25, 2]
		input_path="dtv_to_gamma_199x199.npy"
		save_path="dtv_to_gamma_199x199.png"
	if (True):
		xticks=[0, 48, 98, 148, 198, 248, 298]
		xtick_labels=[2, 50, 100, 150, 200, 250, 300]
		yticks=[0, 48, 98, 148, 198, 248, 298]
		ytick_labels=[300, 250, 200, 150, 100, 50, 2]
		input_path="dtv_to_gamma_299x299.npy"
		save_path="dtv_to_gamma_299x299.png"

	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30
	
	# the image saving settings
	dpi=300
	format="png"

	import numpy as np
	data=np.load(input_path)
	# this will put N to the x-axis and n to the y-axis
	data=data.T
	# this will assure that the final image will have a commonly used orientation
	data=data[::-1, :]
	
	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	plt.ticklabel_format(style="plain")
	plt.imshow(data, cmap="turbo", interpolation="nearest")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.xticks(xticks, xtick_labels)
	plt.yticks(yticks, ytick_labels)
	plt.ylabel("$n$", fontsize=label_fontsize)
	plt.colorbar()
	plt.clim(0, 0.05)
	plt.tight_layout()
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test9():
	verbose=True

	input_dir=MEAN_AND_VAR_FOR_EXACT_DISTRIBUTIONS_DIR
	save_path="experimental_fitting.png"

	from os import listdir
	from os.path import join

	names=sorted(listdir(input_dir))

	import numpy as np
	# set this to True if you want to fit the mean
	if (True):
		target="Mean"
		data_idx=1
		# we use the square root of N for the input to the linear model
		prepare_data=lambda x:np.sqrt(x)
	# otherwise, set it False if you want to fit the variance
	else:
		target="Variance"
		data_idx=2
		# we use the value N for the input to the linear model
		prepare_data=lambda x:x

	# in this example, we load the values with a fixed value of n with N being a free variable
	fixed_n=10

	# in this example the intercept is not fitted, i.e., it is set to zero
	fit_intercept=False

	data=[]
	taker=names
	if (verbose is True):
		print("Collecting the available data...")
		from tqdm import tqdm
		taker=tqdm(taker)
	for name in taker:
		# taking the values of N and n from the file name
		N=int(name.split("_")[1])
		n=int(name.split("_")[3][:name.split("_")[3].find(".")])

		# checking if the fixed value conditions are met
		if (fixed_n is not None and fixed_n!=n):
			continue
		
		input_path=join(input_dir, name)
		with open(input_path, "r", encoding="utf8") as f:
			mean, var=list(map(float, f.read().strip("\r\n").split()))
		data.append((N, mean, var))
	data=sorted(data)
	data=np.array(data)
	
	# prepare the initial data
 	# the values of N are used in both cases
	x=data[:, 0]
	# whether mean of variance is being fitted is determined by the data_idx variable set at the beginning of this function
	y=data[:, data_idx]

	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30

	# the image saving settings
	dpi=300
	format="png"

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	# plotting the ground-truth data
	plt.plot(x, y, label="Ground-truth")

	# preparing the input data
	model_x=np.array([prepare_data(x)]).T
	# a simple linear model is used, although the input data may not necessarily be linear since it depends on the prepare lambda used above
	from sklearn import linear_model
	model=linear_model.LinearRegression()
	model.fit_intercept=fit_intercept
	# here, the model is fitted by using all of the available data, even though skipping the data for lower values of N and/or n may be more useful
	model.fit(model_x, data[:, data_idx])
	# plotting the fitted data
	plt.plot(x, model.predict(model_x), color="green", label="Predicted")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.ylabel(target, fontsize=label_fontsize)

	# printing the model parameters
	print("Coefficients:", model.coef_)
	print("Intercept:", model.intercept_)

	plt.legend()
	plt.tight_layout()
	
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test10():
	input_dir=MEAN_AND_VAR_FOR_EXACT_DISTRIBUTIONS_DIR

	verbose=True

	from os import listdir
	from os.path import join

	names=sorted(listdir(input_dir))

	if (verbose is True):
		from tqdm import tqdm
	
	import numpy as np

	# prepare the configurations for fitting and saving the model parameters
	fitting_and_saving_configurations=[
		(1, lambda x:np.sqrt(x), False, "mean_coefficients.npy", "mean (no intercept)"),
		(1, lambda x:np.sqrt(x), True, "mean_coefficients_and_intercepts.npy", "mean"),
		(2, lambda x:x, False, "var_coefficients.npy", "variance (no intercept)"),
		(2, lambda x:x, True, "var_coefficients_and_intercepts.npy", "variance")
	]

	# determining for which values of n are the models going to be fitted with N being a free variable
	fixed_ns=list(range(1, 500+1))

	# a simple linear model is used, although the input data may not necessarily be linear since it depends on the prepare lambda used above
	from sklearn import linear_model

	# going over the defined configurations
	for data_idx, prepare_data, fit_intercept, output_path, label in fitting_and_saving_configurations:
		if (verbose is True):
			print(label)
		result=dict()
		for fi, fixed_n in enumerate(fixed_ns):
			print("\t%3d / %d   n = %d"%(fi+1, len(fixed_ns), fixed_n))
			data=[]
			taker=names
			if (verbose is True):
				taker=tqdm(taker)
			for name in taker:
				# taking the values of N and n from the file name
				N=int(name.split("_")[1])
				n=int(name.split("_")[3][:name.split("_")[3].find(".")])

				# checking if the fixed value conditions are met
				if (fixed_n is not None and fixed_n!=n):
					continue
				
				input_path=join(input_dir, name)
				with open(input_path, "r", encoding="utf8") as f:
					mean, var=list(map(float, f.read().strip("\r\n").split()))
				data.append((N, mean, var))
			data=sorted(data)
			data=np.array(data)
			
			x=data[:, 0]
			y=data[:, data_idx]

			model_x=np.array([prepare_data(x)]).T
			model=linear_model.LinearRegression()
			model.fit_intercept=fit_intercept
			model.fit(model_x, data[:, data_idx])
			# storing the fitted model parameters
			result[fixed_n]=(*model.coef_, model.intercept_)
		np.save(output_path, result)

def test11():
	# whether to use paramters that were obtained while simultaneously fitting the intercept
	use_data_with_intercept=False

	save_path="additional_experimental_fitting.png"

	# whether to use the intercept in the fitting
	fit_intercept=False

	# this index is set to the index of the model parameter value that we would like to fit - setting to zero means simply using the first parameter, which is in some cases the only one
	data_idx=0

	import numpy as np
	input_path=""
	# set this to True if you want to fit the mean parameter
	if (True):
		target="Parameter for the mean model"
		input_path="mean"
		# we use the square root of N for the input to the linear model
		create_input_for_model=lambda x:[x, x**0.5]
	# otherwise, set it False if you want to fit the variance parameter
	else:
		target="Parameter for the variance model"
		input_path="var"
		# we use the value N for the input to the linear model
		create_input_for_model=lambda x:[x, x**0.5]
	
	input_path+="_coefficients"
	if (use_data_with_intercept is True):
		input_path+="_and_intercepts"
	input_path+=".npy"

	import numpy as np
	data=np.load(input_path, allow_pickle=True).item()

	x=np.array(sorted(data.keys()))
	y=np.array([data[i][data_idx] for i in x])
	
	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30

	# the image saving settings
	dpi=300
	format="png"

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	# plotting the ground-truth data
	plt.plot(x, y, label="Ground-truth")

	model_x=np.array(create_input_for_model(x)).T
	# a simple linear model is used, although the input data may not necessarily be linear since it depends on the prepare lambda used above
	from sklearn import linear_model
	model=linear_model.LinearRegression()
	model.fit_intercept=fit_intercept
	model.fit(model_x, y)
	# plotting the fitted data
	plt.plot(x, model.predict(model_x), color="green", label="Predicted")
	plt.xlabel("$n$", fontsize=label_fontsize)
	plt.ylabel(target, fontsize=label_fontsize)

	# printing the model parameters
	print("Coefficients:", model.coef_)
	print("Intercept:", model.intercept_)

	plt.legend()
	plt.tight_layout()
	
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test12():
	N=500
	n=10

	simulations_count=10000000

	# using None here means that this will not be performed in a multiprocessing manner
	processes_count=8

	verbose=True

	if (verbose is True):
		print("Estimating the mean and the variance...")
	estimated_mean, estimated_var=estimate_mean_and_var_using_monte_carlo(N=N, n=n, simulations_count=simulations_count, processes_count=processes_count, verbose=verbose)

	print("The estimation results:")
	print("\tMean:     %.8f"%(estimated_mean))
	print("\tVariance: %.8f"%(estimated_var))

	mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)
	if (mean is not None and var is not None):
		print()
		print("The ground-truth results:")
		print("\tMean:     %.8f"%(mean))
		print("\tVariance: %.8f"%(var))
		print()
		mean_error=abs(estimated_mean-mean)
		var_error=abs(estimated_var-var)
		print("Mean absolute error:     %.8f, which is %.6f%%"%(mean_error, 100*mean_error/mean))
		print("Variance absolute error: %.8f, which is %.6f%%"%(var_error, 100*var_error/var))

def test13():
	output_dir=ESTIMATED_MEAN_AND_VAR_FOR_SIMULATED_DISTRIBUTIONS_DIR

	#Ns=list(range(1, 10000+1, 1))
	#ns=[10]
	#Ns=list(range(1000+1000*3, 200000+1, 4000))
	#ns=[10]
	#Ns=list(range(2, 200+1, 1))
	#ns=list(range(2, 200+1, 1))
	#Ns=[10, 100, 1000, 10000, 100000, 1000000]
	#ns=[10]
	#Ns=[10**3]
	#ns=list(range(2+7, 200+1, 8))
	#Ns=[10**4]
	#ns=list(range(2+7, 300+1, 8))
	#Ns=[10**4]
	#ns=list(range(2+7, 300+1, 8))
	#Ns=[10**3]
	#ns=list(range(2+7, 300+1, 8))
	#Ns=[500]
	#ns=list(range(2+7, 300+1, 8))
	Ns=[2000]
	ns=list(range(2+7, 300+1, 8))

	#ns=ns[::-1]

	simulations_count=10000000
	processes_count=None

	verbose=True
	overwrite=False

	from os import makedirs
	makedirs(output_dir, exist_ok=True)
	from os.path import isfile, join
	
	combinations=[]
	for N in Ns:
		for n in ns:
			combinations.append((N, n))
	if (verbose is True):
		print("Estimating the mean and the variance...")
	for ci, combination in enumerate(combinations):
		N, n=combination
		if (verbose is True):
			print("%d / %d   N = %d, n = %d"%(ci+1, len(combinations), N, n))
		# prepare the output path
		simlulations_output_dir=join(output_dir, str(simulations_count))
		makedirs(simlulations_output_dir, exist_ok=True)
		output_path=join(simlulations_output_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, n))
		# if overwriting is not included and the result already exists, skip its recalculation
		if (overwrite is False and isfile(output_path) is True):
			continue
		# perform the estimation
		estimated_mean, estimated_var=estimate_mean_and_var_using_monte_carlo(N=N, n=n, simulations_count=simulations_count, processes_count=processes_count, verbose=verbose)
		# save the result
		with open(output_path, "w", encoding="utf8") as fo:
			fo.write(" ".join(map(str, [estimated_mean, estimated_var]))+"\n")

def test14():
	# the values of N
	#Ns=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375]
	Ns=list(range(1, 350+1, 1))
	n=10

	include_fitted_models=True

	save_path="ground-truth_vs_esimated_values.png"

	simulations_count=10000

	import numpy as np
	# set this to True if you want to fit the mean
	if (True):
		target="Mean"
		data_idx=1
		# we use the square root of N for the input to the linear model
		prepare_data=lambda x:np.sqrt(x)
	# otherwise, set it False if you want to fit the variance
	else:
		target="Variance"
		data_idx=2
		# we use the value N for the input to the linear model
		prepare_data=lambda x:x
	
	# in this example the intercept is not fitted, i.e., it is set to zero
	fit_intercept=False

	ground_truth_data=[]
	estimated_data=[]
	for N in Ns:
		mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)
		estimated_mean, estimated_var=load_estimated_mean_and_var_for_simulated_distribution(N=N, n=n, simulations_count=simulations_count)
		ground_truth_data.append((N, mean, var))
		estimated_data.append((N, estimated_mean, estimated_var))
	ground_truth_data=sorted(ground_truth_data)
	ground_truth_data=np.array(ground_truth_data)
	estimated_data=sorted(estimated_data)
	estimated_data=np.array(estimated_data)
	
	# prepare the initial data
 	# the values of N are used in any case
	x=ground_truth_data[:, 0]
	# whether mean of variance is being fitted is determined by the data_idx variable set at the beginning of this function
	y=ground_truth_data[:, data_idx]
	estimated_y=estimated_data[:, data_idx]

	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30

	# the image saving settings
	dpi=300
	format="png"

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	# plotting the ground-truth data
	plt.plot(x, y, label="Ground-truth")
	# plotting the ground-truth data
	plt.plot(x, estimated_y, label="Estimation")

	if (include_fitted_models is True):
		# preparing the input data
		model_x=np.array([prepare_data(x.astype(np.float32))]).T
		# a simple linear model is used, although the input data may not necessarily be linear since it depends on the prepare lambda used above
		from sklearn import linear_model
		model=linear_model.LinearRegression()
		model.fit_intercept=fit_intercept
		# here, the model is fitted by using all of the available data, even though skipping the data for lower values of N and/or n may be more useful
		model.fit(model_x, ground_truth_data[:, data_idx])
		# plotting the fitted data
		plt.plot(x, model.predict(model_x), color="green", label="Predicted on ground-truth values")
		plt.xlabel("$N$", fontsize=label_fontsize)
		plt.ylabel(target, fontsize=label_fontsize)

		# repeating the procedure above for the estimated values
		model=linear_model.LinearRegression()
		model.fit_intercept=fit_intercept
		# here, the model is fitted by using all of the available data, even though skipping the data for lower values of N and/or n may be more useful
		model.fit(model_x, estimated_data[:, data_idx])
		# plotting the fitted data
		plt.plot(x, model.predict(model_x), color="red", label="Predicted on estimated values")
		plt.xlabel("$N$", fontsize=label_fontsize)
		plt.ylabel(target, fontsize=label_fontsize)

	plt.legend()
	plt.tight_layout()
	
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test15():
	verbose=True

	simulations_count=1000

	Ns=list(range(2, 200+1))
	ns=list(range(2, 200+1))
	output_path="dtv_cdf_0.95_bin_mc_%d_199x199.npy"%simulations_count

	from scipy.stats import gamma
	
	results=dict()
	taker=Ns
	if (verbose is True):
		print("Assessing the accuracy of using the gamma distribution with estimated parameters to approximate the DTV distribution...")
		from tqdm import tqdm
		taker=tqdm(taker)
	from tqdm import tqdm
	# go over the values of N
	for N in taker:
		# go over the values of n
		for n in ns:
			
			distribution=load_exact_dtv_distribution(N=N, n=n)

			mean, var=load_estimated_mean_and_var_for_simulated_distribution(N=N, n=n, simulations_count=simulations_count)

			gamma_alpha=(mean**2)/var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=var/mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			# this sum will be used to calculate the exact values of CDF
			summed=sum([dtv_count for dtv, dtv_count in distribution])

			# calculate the mean absolute error of CDF for every value from 0 to 2*N
			score=0
			count=0
			cummulative=0
			for i in range(0, 2*N+1):
				cummulative+=distribution[i][1]
				exact_value=cummulative/summed
				approximated_value=g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				score+=abs(approximated_value-exact_value)
				count+=1
			score/=count
			results[(N, n)]=score

	# prepare the results for saving
	import numpy as np
	data=[]
	for N in Ns:
		current=[]
		for n in ns:
			current.append(results[(N, n)])
		data.append(current)
	data=np.array(data)
	# save the obtained results
	np.save(output_path, data)

def test16():
	verbose=True
	
	# setting simulations_count to None will result in using the ground-truth parameter values
	simulations_count=None

	Ns=list(range(2, 200+1))
	ns=list(range(2, 200+1))
	description="199x199"

	cdf_threshold=0.95
	
	if (simulations_count is None):
		output_path="dtv_cdf_%s_bin_gamma_exact_%s.npy"%(str(cdf_threshold), description)
	else:
		output_path="dtv_cdf_%s_bin_gamma_mc_%d_%s.npy"%(str(cdf_threshold), simulations_count, description)

	from scipy.stats import gamma
	
	results=dict()
	taker=Ns
	if (verbose is True):
		print("Calculating the bins for which the used gamma distribution claims a CDF over the given threshold...")
		from tqdm import tqdm
		taker=tqdm(taker)
	from tqdm import tqdm
	# go over the values of N
	for N in taker:
		# go over the values of n
		for n in ns:
			if (simulations_count is not None):
				mean, var=load_estimated_mean_and_var_for_simulated_distribution(N=N, n=n, simulations_count=simulations_count)
			else:
				mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)

			gamma_alpha=(mean**2)/var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=var/mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			
			# look for the bin with the CDF over the given threshold
			target_bin=None
			i=0
			approximated_value=0
			while(True):
				previous_approximated_value=approximated_value
				approximated_value=g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				if (approximated_value>=cdf_threshold):
					target_bin=i
					# check if the previous approximated cummulative probability is closer to the desired threshold
					if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
						target_bin-=1
					break
				i+=1
			results[(N, n)]=target_bin

	# prepare the results for saving
	import numpy as np
	data=[]
	for N in Ns:
		current=[]
		for n in ns:
			current.append(results[(N, n)])
		data.append(current)
	data=np.array(data)
	# save the obtained results
	np.save(output_path, data)

def test17():
	Ns=list(range(2, 200+1))
	ns=list(range(2, 200+1))

	# setting simulations_count to None will result in using the ground-truth parameter values
	simulations_count1=None
	simulations_count2=10000

	description="199x199"

	cdf_threshold=0.95
	
	if (simulations_count1 is None):
		input_path1="dtv_cdf_%s_bin_gamma_exact_%s.npy"%(str(cdf_threshold), description)
	else:
		input_path1="dtv_cdf_%s_bin_gamma_mc_%d_%s.npy"%(str(cdf_threshold), simulations_count1, description)
	if (simulations_count2 is None):
		input_path2="dtv_cdf_%s_bin_gamma_exact_%s.npy"%(str(cdf_threshold), description)
	else:
		input_path2="dtv_cdf_%s_bin_gamma_mc_%d_%s.npy"%(str(cdf_threshold), simulations_count2, description)
	
	import numpy as np
	data1=np.load(input_path1, allow_pickle=True)
	data2=np.load(input_path2, allow_pickle=True)

	count_all=0
	score_all=0
	score_all_err_1=0

	score_N_greater_than_or_equal_to_n_score=0
	count_N_greater_than_or_equal_to_n_score=0
	score_N_greater_than_or_equal_to_n_score_err_1=0

	score_N_greater_than_or_equal_to_5n_score=0
	count_N_greater_than_or_equal_to_5n_score=0
	score_N_greater_than_or_equal_to_5n_score_err_1=0
	
	for Ni, N in enumerate(Ns):
		for ni, n in enumerate(ns):
			bin1=data1[Ni][ni]
			bin2=data2[Ni][ni]

			count_all+=1
			if (bin1==bin2):
				score_all+=1
			elif (abs(bin1-bin2)==1):
				score_all_err_1+=1
			
			if (N>=n):
				count_N_greater_than_or_equal_to_n_score+=1
				if (bin1==bin2):
					score_N_greater_than_or_equal_to_n_score+=1
				elif (abs(bin1-bin2)==1):
					score_N_greater_than_or_equal_to_n_score_err_1+=1
			
			if (N>=5*n):
				count_N_greater_than_or_equal_to_5n_score+=1
				if (bin1==bin2):
					score_N_greater_than_or_equal_to_5n_score+=1
				elif (abs(bin1-bin2)==1):
					score_N_greater_than_or_equal_to_5n_score_err_1+=1
	
	print()

	print("The bins are the same in %.2f%% of all cases."%(100*score_all/count_all))
	print("The bins are the same in %.2f%% of cases where N >= n."%(100*score_N_greater_than_or_equal_to_n_score/count_N_greater_than_or_equal_to_n_score))
	print("The bins are the same in %.2f%% of cases where N >= 5*n."%(100*score_N_greater_than_or_equal_to_5n_score/count_N_greater_than_or_equal_to_5n_score))

	print()

	print("The bins are within distance of 1 in %.2f%% of all cases."%(100*(score_all+score_all_err_1)/count_all))
	print("The bins are within distance of 1 in %.2f%% of cases where N >= n."%(100*(score_N_greater_than_or_equal_to_n_score+score_N_greater_than_or_equal_to_n_score_err_1)/count_N_greater_than_or_equal_to_n_score))
	print("The bins are within distance of 1 in %.2f%% of cases where N >= 5*n."%(100*(score_N_greater_than_or_equal_to_5n_score+score_N_greater_than_or_equal_to_5n_score_err_1)/count_N_greater_than_or_equal_to_5n_score))
	
	print()

def test18():
	# the fitting will be tested on this data point
	target_N=200

	target_n=10

	# determining which data points should be used for fitting
	fitting_Ns=[25, 50, 75, 100, 125, 150]
	#fitting_Ns=[10000]
	fitting_n=10

	simulations_count=10000

	cdf_threshold=0.95

	# functions for preparing the data that will be loaded and used for modelling
	import numpy as np
	prepare_mean_data=lambda x:np.sqrt(x)
	prepare_var_data=lambda x:x

	fit_intercept=False

	# loading the data required for fitting
	mean_x=[]
	mean_y=[]
	var_x=[]
	var_y=[]
	for fitting_N in fitting_Ns:
		if (simulations_count is None):
			mean, var=load_mean_and_var_for_exact_distribution(N=fitting_N, n=fitting_n)
		else:
			mean, var=load_estimated_mean_and_var_for_simulated_distribution(N=fitting_N, n=fitting_n, simulations_count=simulations_count)
		mean_x.append(fitting_N)
		var_x.append(fitting_N)
		mean_y.append(mean)
		var_y.append(var)
	
	mean_x=np.array(mean_x)
	var_x=np.array(var_x)
	mean_y=np.array(mean_y)
	var_y=np.array(var_y)

	mean_x=np.array([prepare_mean_data(mean_x)]).T
	var_x=np.array([prepare_var_data(var_x)]).T
	# a simple linear model is used, although the input data may not necessarily be linear since it depends on the prepare lambda used above
	from sklearn import linear_model
	mean_model=linear_model.LinearRegression()
	mean_model.fit_intercept=fit_intercept
	var_model=linear_model.LinearRegression()
	var_model.fit_intercept=fit_intercept
	# here, the model is fitted by using all of the available data, even though skipping the data for lower values of N and/or n may be more useful
	mean_model.fit(mean_x, mean_y)
	var_model.fit(var_x, var_y)

	# first, the mean and variance are checked

	print()
	print("N = "+str(target_N))
	print("n = "+str(target_n))
	print()

	print("GAMMA DISTRIBUTION")
	# check for the ground-truth data
	gt_mean, gt_var=load_mean_and_var_for_exact_distribution(N=target_N, n=target_n)
	if (gt_mean is not None and gt_var is not None):
		print()
		print("Ground-truth mean:    ", gt_mean)
		print("Ground-truth variance:", gt_var)
	# check for the estimated data
	estimated_mean, estimated_var=load_estimated_mean_and_var_for_simulated_distribution(N=target_N, n=target_n, simulations_count=simulations_count)
	if (estimated_mean is not None and estimated_var is not None):
		print()
		print("Estimated mean:       ", estimated_mean)
		print("Estimated variance:   ", estimated_var)
	# check for the modelled data
	modelled_mean=mean_model.predict(np.array([prepare_mean_data(np.array([target_N]))]).T)[0]
	modelled_var=var_model.predict(np.array([prepare_var_data(np.array([target_N]))]).T)[0]
	print()
	print("Modelled mean:        ", modelled_mean)
	print("Modelled variance:    ", modelled_var)
	
	print()
	print()
	print()

	from scipy.stats import gamma

	print("THRESHOLD DTV FOR "+str(cdf_threshold))
	# next, the threshold DTV's are checked
	distribution=load_exact_dtv_distribution(N=target_N, n=target_n)
	if (distribution is not None):
		# preparing the numerator and denominator for easier dealing with precision
		numerator=cdf_threshold
		denominator=1
		while(numerator!=int(numerator)):
			numerator*=10
			denominator*=10
		numerator=int(numerator)
		# calculate the components required for calculating the mean value
		whole_count=target_n**target_N
		prepared_target_count=numerator*whole_count
		gt_bin=None
		count=0
		for dtv, dtv_count in distribution:
			previous_count=count
			count+=dtv_count
			if (denominator*count>=prepared_target_count):
				gt_bin=dtv
				if (abs(prepared_target_count-denominator*previous_count)<abs(prepared_target_count-denominator*count)):
					gt_bin-=1
				break
		print()
		print("Ground-truth DTV:      ", gt_bin)
	
	# check for the ground-truth gamma-distribution data
	if (gt_mean is not None and gt_var is not None):
		gamma_alpha=(gt_mean**2)/gt_var
		# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
		gamma_beta=gt_var/gt_mean
		gamma_loc=0
		
		# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
		g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
		gt_gamma_bin=0
		approximated_value=0
		while(True):
			previous_approximated_value=approximated_value
			approximated_value=g.cdf(gt_gamma_bin+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
			if (approximated_value>=cdf_threshold):
				# check if the previous approximated cummulative probability is closer to the desired threshold
				if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
					gt_gamma_bin-=1
				break
			gt_gamma_bin+=1
		print()
		print("Ground-truth gamma DTV:", gt_gamma_bin)
	
	# check for the estimated gamma-distribution data
	if (estimated_mean is not None and estimated_var is not None):
		gamma_alpha=(estimated_mean**2)/estimated_var
		# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
		gamma_beta=estimated_var/estimated_mean
		gamma_loc=0
		
		# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
		g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
		estimated_gamma_bin=0
		approximated_value=0
		while(True):
			previous_approximated_value=approximated_value
			approximated_value=g.cdf(estimated_gamma_bin+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
			if (approximated_value>=cdf_threshold):
				# check if the previous approximated cummulative probability is closer to the desired threshold
				if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
					estimated_gamma_bin-=1
				break
			estimated_gamma_bin+=1
		print()
		print("Estimated gamma DTV:   ", estimated_gamma_bin)
	
	# check for the modelled gamma-distribution data
	if (modelled_mean is not None and modelled_var is not None):
		gamma_alpha=(modelled_mean**2)/modelled_var
		# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
		gamma_beta=modelled_var/modelled_mean
		gamma_loc=0
		
		# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
		g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
		modelled_gamma_bin=0
		approximated_value=0
		while(True):
			previous_approximated_value=approximated_value
			approximated_value=g.cdf(modelled_gamma_bin+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
			if (approximated_value>=cdf_threshold):
				# check if the previous approximated cummulative probability is closer to the desired threshold
				if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
					modelled_gamma_bin-=1
				break
			modelled_gamma_bin+=1
		print()
		print("Modelled gamma DTV:    ", modelled_gamma_bin)

	print()

def test19():
	# the fitting will be tested on this data point
	target_Ns=list(range(2, 400+1, 1))
	target_n=10

	verbose=True

	# determining which data points should be used for fitting
	fitting_Ns=[25, 150, 1000]
	#fitting_Ns=[10000]
	fitting_n=10

	simulations_count=10000

	cdf_threshold=0.95

	# functions for preparing the data that will be loaded and used for modelling
	import numpy as np
	prepare_mean_data=lambda x:np.sqrt(x)
	prepare_var_data=lambda x:x

	fit_intercept=False

	# loading the data required for fitting
	mean_x=[]
	mean_y=[]
	var_x=[]
	var_y=[]
	for fitting_N in fitting_Ns:
		if (simulations_count is None):
			mean, var=load_mean_and_var_for_exact_distribution(N=fitting_N, n=fitting_n)
		else:
			mean, var=load_estimated_mean_and_var_for_simulated_distribution(N=fitting_N, n=fitting_n, simulations_count=simulations_count)
		mean_x.append(fitting_N)
		var_x.append(fitting_N)
		mean_y.append(mean)
		var_y.append(var)
	
	mean_x=np.array(mean_x)
	var_x=np.array(var_x)
	mean_y=np.array(mean_y)
	var_y=np.array(var_y)

	mean_x=np.array([prepare_mean_data(mean_x)]).T
	var_x=np.array([prepare_var_data(var_x)]).T
	# a simple linear model is used, although the input data may not necessarily be linear since it depends on the prepare lambda used above
	from sklearn import linear_model
	mean_model=linear_model.LinearRegression()
	mean_model.fit_intercept=fit_intercept
	var_model=linear_model.LinearRegression()
	var_model.fit_intercept=fit_intercept
	# here, the model is fitted by using all of the available data, even though skipping the data for lower values of N and/or n may be more useful
	mean_model.fit(mean_x, mean_y)
	var_model.fit(var_x, var_y)

	from scipy.stats import gamma

	taker=target_Ns
	if (verbose is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	
	gt_means=[]
	gt_vars=[]
	estimated_means=[]
	estimated_vars=[]
	modelled_means=[]
	modelled_vars=[]

	gt_bins=[]
	gt_gamma_bins=[]
	estimated_gamma_bins=[]
	modelled_gamma_bins=[]
	for target_N in taker:
		# check for the ground-truth data
		gt_mean, gt_var=load_mean_and_var_for_exact_distribution(N=target_N, n=target_n)
		estimated_mean, estimated_var=load_estimated_mean_and_var_for_simulated_distribution(N=target_N, n=target_n, simulations_count=simulations_count)
		modelled_mean=mean_model.predict(np.array([prepare_mean_data(np.array([target_N]))]).T)[0]
		modelled_var=var_model.predict(np.array([prepare_var_data(np.array([target_N]))]).T)[0]

		gt_means.append(gt_mean)
		gt_vars.append(gt_var)
		estimated_means.append(estimated_mean)
		estimated_vars.append(estimated_var)
		modelled_means.append(modelled_mean)
		modelled_vars.append(modelled_var)

		# next, the threshold DTV's are checked
		distribution=load_exact_dtv_distribution(N=target_N, n=target_n)
		if (distribution is not None):
			# preparing the numerator and denominator for easier dealing with precision
			numerator=cdf_threshold
			denominator=1
			while(numerator!=int(numerator)):
				numerator*=10
				denominator*=10
			numerator=int(numerator)
			# calculate the components required for calculating the mean value
			whole_count=target_n**target_N
			prepared_target_count=numerator*whole_count
			gt_bin=None
			count=0
			for dtv, dtv_count in distribution:
				previous_count=count
				count+=dtv_count
				if (denominator*count>=prepared_target_count):
					gt_bin=dtv
					if (abs(prepared_target_count-denominator*previous_count)<abs(prepared_target_count-denominator*count)):
						gt_bin-=1
					break
			gt_bins.append(gt_bin)
		
		# check for the ground-truth gamma-distribution data
		if (gt_mean is not None and gt_var is not None):
			gamma_alpha=(gt_mean**2)/gt_var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=gt_var/gt_mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			gt_gamma_bin=0
			approximated_value=0
			while(True):
				previous_approximated_value=approximated_value
				approximated_value=g.cdf(gt_gamma_bin+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				if (approximated_value>=cdf_threshold):
					# check if the previous approximated cummulative probability is closer to the desired threshold
					if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
						gt_gamma_bin-=1
					break
				gt_gamma_bin+=1
			gt_gamma_bins.append(gt_gamma_bin)
		
		# check for the estimated gamma-distribution data
		if (estimated_mean is not None and estimated_var is not None):
			gamma_alpha=(estimated_mean**2)/estimated_var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=estimated_var/estimated_mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			estimated_gamma_bin=0
			approximated_value=0
			while(True):
				previous_approximated_value=approximated_value
				approximated_value=g.cdf(estimated_gamma_bin+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				if (approximated_value>=cdf_threshold):
					# check if the previous approximated cummulative probability is closer to the desired threshold
					if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
						estimated_gamma_bin-=1
					break
				estimated_gamma_bin+=1
			estimated_gamma_bins.append(estimated_gamma_bin)
		
		# check for the modelled gamma-distribution data
		if (modelled_mean is not None and modelled_var is not None):
			gamma_alpha=(modelled_mean**2)/modelled_var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=modelled_var/modelled_mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			modelled_gamma_bin=0
			approximated_value=0
			while(True):
				previous_approximated_value=approximated_value
				approximated_value=g.cdf(modelled_gamma_bin+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				if (approximated_value>=cdf_threshold):
					# check if the previous approximated cummulative probability is closer to the desired threshold
					if (abs(cdf_threshold-previous_approximated_value)<abs(cdf_threshold-approximated_value)):
						modelled_gamma_bin-=1
					break
				modelled_gamma_bin+=1
			modelled_gamma_bins.append(modelled_gamma_bin)
	
	gt_bins=np.array(gt_bins)
	gt_gamma_bins=np.array(gt_gamma_bins)
	estimated_gamma_bins=np.array(estimated_gamma_bins)
	modelled_gamma_bins=np.array(modelled_gamma_bins)
	
	print()
	print("Full match statistics:")
	print()
	print("Ground-truth gamma bins accuracy: %.2f%%"%(100*sum(gt_gamma_bins==gt_bins)/len(gt_bins)))
	print("Estimated gamma bins accuracy:    %.2f%%"%(100*sum(estimated_gamma_bins==gt_bins)/len(gt_bins)))
	print("Modelled gamma bins accuracy:     %.2f%%"%(100*sum(modelled_gamma_bins==gt_bins)/len(gt_bins)))
	print()
	print()
	print()
	print("Matching up to 1 statistics:")
	print()
	print("Ground-truth gamma bins accuracy: %.2f%%"%(100*sum(np.abs(gt_gamma_bins-gt_bins)<=1)/len(gt_bins)))
	print("Estimated gamma bins accuracy:    %.2f%%"%(100*sum(np.abs(estimated_gamma_bins-gt_bins)<=1)/len(gt_bins)))
	print("Modelled gamma bins accuracy:     %.2f%%"%(100*sum(np.abs(modelled_gamma_bins-gt_bins)<=1)/len(gt_bins)))
	print()

def test20():
	N=200
	n=10

	distribution=load_exact_dtv_distribution(N=N, n=n)
	mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)

	correction=CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA
	upper_dtv=2*N
	#correction=0.5
	#upper_dtv=20

	from scipy.stats import gamma

	gamma_alpha=(mean**2)/var
	# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
	gamma_beta=var/mean
	gamma_loc=0
	
	# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
	g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
	# this sum will be used to calculate the exact values of CDF
	summed=sum([dtv_count for dtv, dtv_count in distribution])

	# calculate the mean absolute error of CDF for every value from 0 to 2*N
	cummulative=0
	for i in range(0, upper_dtv+1):
		cummulative+=distribution[i][1]
		exact_point_value=distribution[i][1]/summed
		approximated_point_value=g.cdf(i+correction)-g.cdf(i-1+correction)
		exact_cummulative_value=cummulative/summed
		approximated_cummulative_value=g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
		print("%3d:   %.8f %.8f   %.8f %.8f"%(i, exact_point_value, approximated_point_value, exact_cummulative_value, approximated_cummulative_value))

def test21():
	N=20
	n=4

	save_path="dtv_distribution_histogram_N_%d_n_%d.png"%(N, n)

	show=True

	format="png"
	dpi=300
	figsize=(13, 8)
	tick_fontsize=22
	label_fontsize=36

	xlabel="DTV"
	ylabel="Appearance"

	distribution=load_exact_dtv_distribution(N=N, n=n)
	
	quantities=[b for a, b in distribution]
	bins=[a for a, b in distribution]
	#bins+=[max(bins)+1]
	
	import matplotlib
	import matplotlib.pyplot as plt
	
	matplotlib.rc("font", size=tick_fontsize)
	fig=plt.figure(figsize=figsize)
	ax=fig.add_subplot(1, 1, 1)
	ax.bar(bins, height=quantities, width=1.0, align="center", edgecolor="black")
	plt.ticklabel_format(style="plain")
	from matplotlib.ticker import ScalarFormatter
	ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
	
	#font={"family":"normal", "weight":"bold", "size":tick_fontsize};
	plt.xlabel(xlabel, fontsize=label_fontsize)
	plt.ylabel(ylabel, fontsize=label_fontsize)
	
	#plt.xticks(selected_years, list(map(year_lambda, selected_years)));
	
	plt.tight_layout()
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	
	if (show==True):
		plt.show()
	
	plt.clf()

def test22():
	N=50
	n=10

	save_path="dtv_distribution_histogram_and_gamma_N_%d_n_%d.png"%(N, n)
	
	show=True

	format="png"
	dpi=300
	figsize=(13, 8)
	tick_fontsize=22
	label_fontsize=36

	xlabel="DTV"
	ylabel="PDF"

	distribution=load_exact_dtv_distribution(N=N, n=n)
	mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)
	
	gamma_alpha=(mean**2)/var
	gamma_beta=var/mean
	gamma_loc=0
	
	from scipy.stats import gamma
	g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
	
	quantities=[b for a, b in distribution]
	bins=[a for a, b in distribution]
	#bins+=[max(bins)+1]
	
	import matplotlib
	import matplotlib.pyplot as plt
	
	matplotlib.rc("font", size=tick_fontsize)
	fig=plt.figure(figsize=figsize)
	ax=fig.add_subplot(1, 1, 1)
	import numpy as np
	ax.bar(bins, height=np.array(quantities)/np.sum(quantities), width=1.0, align="center", edgecolor="black")

	x=np.linspace(0, 2*N, 1000)
	y=g.pdf(x)
	ax.plot(x+0.5, y, "-", lw=2, label="fitted gamma distribution", color="red")
	

	plt.ticklabel_format(style="plain")
	from matplotlib.ticker import ScalarFormatter
	ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
	
	#font={"family":"normal", "weight":"bold", "size":tick_fontsize};
	plt.xlabel(xlabel, fontsize=label_fontsize)
	plt.ylabel(ylabel, fontsize=label_fontsize)
	
	#plt.xticks(selected_years, list(map(year_lambda, selected_years)));
	
	plt.legend()
	plt.tight_layout()
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	
	if (show==True):
		plt.show()
	
	plt.clf()

def test23():

	repeat=1000000
	Ns=list(range(2, 400+1))
	#Ns=list(range(25, 1000+1, 25))
	#Ns=list(range(100, 40000+1, 100))
	Ns=list(range(1000+1000*4, 200000+1, 1000*5))
	n=10

	d=1

	from os.path import isfile, join
	output_dir=join(OBTAINED_P_AND_DTV_VALUES_DIR, str(repeat))
	from os import makedirs

	makedirs(output_dir, exist_ok=True)

	import numpy as np
	ps=np.zeros((n,))
	
	ps[::2]=10**(d+1)+1
	ps[1::2]=10**(d+1)-1
	
	from scipy.stats import chisquare
	
	from tqdm import tqdm
	for Ni, N in enumerate(Ns):
		output_path=join(output_dir, "round_half_to_even_d_%d_N_%d_n_%d.txt"%(d, N, n))
		print("%d / %d   N = %d"%(Ni+1, len(Ns), N))
		if (isfile(output_path) is True):
			continue
		chisquare_ps=[]
		dtvs=[]
		for i in tqdm(range(repeat)):
			bins=get_random_sample_histogram(ps=ps, size=N)
			dtv=get_dtv(bins)
			_, chisquare_p=chisquare(bins)
			dtvs.append(dtv)
			chisquare_ps.append(chisquare_p)
	
		with open(output_path, "w", encoding="utf8") as fo:
			for chisquare_p, dtv in zip(chisquare_ps, dtvs):
				fo.write(" ".join(map(str, [chisquare_p, dtv]))+"\n")

def test24():
	#input_path="CHF_vs_EUR.csv"
	#input_path="USD_vs_EUR.csv"
	input_path="CNY_vs_EUR.csv"

	save_path=None

	verbose=True
	
	from scipy.stats import chisquare

	n=10
	simulations_count=10**6

	format="png"
	dpi=300
	figsize=(13, 8)
	tick_fontsize=22
	label_fontsize=36

	xlabel="$p$ value"
	ylabel="Observed last digits"


	lines=load_lines(input_path)
	lines=lines[1:]
	last_digits=[0]*n

	chisquare_ps=[]
	dtv_ps=[]
	gamma_ps=[]
	Ns=[]
	taker=lines
	if (verbose is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	for l in taker:
		parts=l.split(",")
		if (len(parts)==3):
			digit=int(parts[-1].replace("\"", "")[-1])
			last_digits[digit]+=1
		else:
			continue
		
		_, chisquare_p=chisquare(last_digits)
		chisquare_ps.append(chisquare_p)
		
		dtv=get_dtv(last_digits)
		N=sum(last_digits)
		Ns.append(N)
		distribution=load_exact_dtv_distribution(N=N, n=n)
		if (distribution is not None):
			s=0
			for i in range(0, dtv-1+1):
				s+=distribution[i][1]
			dtv_p=1-s/(n**N)
			dtv_ps.append(dtv_p)
		mean, var=load_estimated_mean_and_var_for_simulated_distribution(N=N, n=n, simulations_count=simulations_count)
		if (mean is not None and var is not None):
			gamma_alpha=(mean**2)/var
			gamma_beta=var/mean
			gamma_loc=0
			
			from scipy.stats import gamma
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			gamma_p=1-g.cdf(dtv+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
			gamma_ps.append(gamma_p)
			
	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	#plt.ticklabel_format(style="plain")
	plt.xlabel(xlabel, fontsize=label_fontsize)
	#plt.xticks(xticks, xtick_labels)
	#plt.yticks(yticks, ytick_labels)
	plt.ylabel(ylabel, fontsize=label_fontsize)
	#plt.colorbar()
	#plt.clim(0, 0.05)
	plt.plot(Ns[:len(chisquare_ps)], chisquare_ps, label="Pearson's chi-square test $p$-value")
	plt.plot(Ns[:len(dtv_ps)], dtv_ps, label="CT $p$-value")
	plt.plot(Ns[:len(gamma_ps)], gamma_ps, label="CT gamma $p$-value")
	plt.plot(Ns, [0.05]*len(Ns), "--", color="red", label="$p=0.05$")
	plt.legend()
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test25():
	repeat=1000000
	simulations_count=1000000
	d=1

	step=1000
	
	verbose=False

	from os.path import isfile, join
	input_dir=join(OBTAINED_P_AND_DTV_VALUES_DIR, str(repeat))

	output_path="mean_p_values_d_%d_step_%d.txt"%(d, step)

	previous=None
	if (isfile(output_path) is True):
		lines=load_lines(output_path)
		previous=dict()
		for l in lines:
			previous[l.split()[0]]=l
	
	#Ns=list(range(2, 300+1))
	#Ns=list(range(2, 99+1))+list(range(100, 4000+1, step))
	#Ns=list(range(2, 99+1))+list(range(100, 1000+1, 100))+list(range(1000, 45000+1, step))
	Ns=list(range(2, 160000+1))
	#Ns=list(range(1000, 5000+1, 1000))
	n=10

	import numpy as np
	from scipy.stats import gamma

	taker=list(enumerate(Ns))
	if (verbose is True):
		from tqdm import tqdm
		taker=tqdm(taker)
	with open(output_path, "w", encoding="utf8") as fo:
		for Ni, N in taker:
			input_path=join(input_dir, "round_half_to_even_d_%d_N_%d_n_%d.txt"%(d, N, n))
			
			if (isfile(input_path) is False):
				continue

			if (previous is not None and str(N) in previous.keys()):
				print(previous[str(N)])
				fo.write(previous[str(N)]+"\n")
				fo.flush()
				continue

			distribution=load_exact_dtv_distribution(N=N, n=n)
			dtv_ps_table=[]
			if (distribution is not None):
				if (distribution is not None):
					s=0
					for i in range(0, 2*N+1):
						dtv_p=1-s/(n**N)
						dtv_ps_table.append(dtv_p)
						s+=distribution[i][1]
			else:
				mean, var=load_estimated_mean_and_var_for_simulated_distribution(N, n, simulations_count)
				if (mean is None or var is None):
					continue
				gamma_alpha=(mean**2)/var
				gamma_beta=var/mean
				gamma_loc=0
				g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
				dtv_p=0
				for i in range(0, 2*N+1):
					dtv_ps_table.append(dtv_p)
					dtv_p=1-g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
					
			data=np.loadtxt(input_path)
			chisquare_ps=data[:, 0]
			dtv_ps=np.array([dtv_ps_table[int(x)] for x in data[:, 1]])
			print(N, np.mean(chisquare_ps), np.mean(dtv_ps))
			fo.write(" ".join(map(str, [N, np.mean(chisquare_ps), np.mean(dtv_ps)]))+"\n")
			fo.flush()
		
def test26():
	if (True):
		d=0
		step=100
		lower_N=2
		upper_N=3000
	if (True):
		d=1
		step=1000
		lower_N=2
		upper_N=160000
	
	input_path="mean_p_values_d_%d_step_%d.txt"%(d, step)
	save_path="mean_p_values_d_%d_step_%d.png"%(d, step)

	import numpy as np
	data=np.loadtxt(input_path)

	Ns=data[:, 0]
	data=data[Ns<=upper_N, :]
	Ns=data[:, 0]
	data=data[Ns>=lower_N, :]
	mean_chisquare_ps=data[:, 1]
	mean_dtv_ps=data[:, 2]

	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30

	# the image saving settings
	dpi=300
	format="png"

	xlabel="$N$"
	ylabel="$p$ value"

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	#plt.ticklabel_format(style="plain")
	plt.xlabel(xlabel, fontsize=label_fontsize)
	#plt.xticks(xticks, xtick_labels)
	#plt.yticks(yticks, ytick_labels)
	plt.ylabel(ylabel, fontsize=label_fontsize)
	#plt.colorbar()
	#plt.clim(0, 0.05)
	plt.plot(Ns, mean_chisquare_ps, label="mean Pearson's chi-square test $p$ value")
	plt.plot(Ns, mean_dtv_ps, label="mean CT gamma $p$ value")
	plt.plot(Ns, [0.05]*len(Ns), "--", color="red", label="$p=0.05$")
	plt.legend()
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test27():
	verbose=True
	
	# setting simulations_count to None will result in using the ground-truth parameter values
	simulations_count=None

	if (False):
		Ns=list(range(2, 200+1))
		ns=list(range(2, 200+1))
		description="199x199"
	if (True):
		Ns=list(range(2, 300+1))
		ns=list(range(2, 300+1))
		description="299x299"

	p_threshold=0.05
	
	if (simulations_count is None):
		output_path="dtv_p_%s_bin_gamma_exact_%s.npy"%(str(p_threshold), description)
	else:
		output_path="dtv_p_%s_bin_gamma_mc_%d_%s.npy"%(str(p_threshold), simulations_count, description)

	from scipy.stats import gamma
	
	results=dict()
	taker=Ns
	if (verbose is True):
		print("Calculating the bins for which the used gamma distribution claims a p closest to the given threshold...")
		from tqdm import tqdm
		taker=tqdm(taker)
	from tqdm import tqdm
	# go over the values of N
	for N in taker:
		# go over the values of n
		for n in ns:
			if (simulations_count is not None):
				mean, var=load_estimated_mean_and_var_for_simulated_distribution(N=N, n=n, simulations_count=simulations_count)
			else:
				mean, var=load_mean_and_var_for_exact_distribution(N=N, n=n)

			gamma_alpha=(mean**2)/var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=var/mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			
			# look for the bin with the CDF over the given threshold
			target_bin=None
			i=0
			approximated_value=1
			previous_approximated_value=None
			while(True):
				if (approximated_value<=p_threshold):
					target_bin=i
					# check if the previous approximated p is closer to the desired threshold
					if (previous_approximated_value is not None and abs(p_threshold-previous_approximated_value)<abs(p_threshold-approximated_value)):
						target_bin-=1
					break
				previous_approximated_value=approximated_value
				approximated_value=1-g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				i+=1
			if (target_bin is None):
				target_bin=2*N
			results[(N, n)]=target_bin

	# prepare the results for saving
	import numpy as np
	data=[]
	for N in Ns:
		current=[]
		for n in ns:
			current.append(results[(N, n)])
		data.append(current)
	data=np.array(data)
	# save the obtained results
	np.save(output_path, data)

def test28():
	verbose=True
	
	if (False):
		Ns=list(range(2, 200+1))
		ns=list(range(2, 200+1))
		description="199x199"
	if (True):
		Ns=list(range(2, 300+1))
		ns=list(range(2, 300+1))
		description="299x299"

	p_threshold=0.05
	
	output_path="dtv_p_%s_bin_exact_%s.npy"%(str(p_threshold), description)

	results=dict()
	taker=Ns
	if (verbose is True):
		print("Calculating the bins for which the used gamma distribution claims a p closest to the given threshold...")
		from tqdm import tqdm
		taker=tqdm(taker)
	from tqdm import tqdm
	# go over the values of N
	for N in taker:
		# go over the values of n
		for n in ns:
			distribution=load_exact_dtv_distribution(N=N, n=n)

			s=0
			target_bin=None
			dtv_p=1
			previous_dtv_p=None
			for i in range(0, 2*N+1):
				if (dtv_p<p_threshold):
					target_bin=i
					if (previous_dtv_p is not None and abs(p_threshold-previous_dtv_p)<abs(p_threshold-dtv_p)):
						target_bin-=1
					break
				s+=distribution[i][1]
				previous_dtv_p=dtv_p
				dtv_p=1-s/(n**N)
			if (target_bin is None):
				target_bin=2*N
			results[(N, n)]=target_bin

	# prepare the results for saving
	import numpy as np
	data=[]
	for N in Ns:
		current=[]
		for n in ns:
			current.append(results[(N, n)])
		data.append(current)
	data=np.array(data)
	# save the obtained results
	np.save(output_path, data)

def test29():
	verbose=True
	
	# setting simulations_count to None will result in using the ground-truth parameter values
	simulations_count=10**7

	N_for_fitting=500
	#N_for_fitting=10**3
	#N_for_fitting=10**4

	fit_intercept=False
	
	if (True):
		Ns=list(range(2, 200+1))
		ns=list(range(2, 200+1))
		description="199x199"
	if (False):
		Ns=list(range(2, 300+1))
		ns=list(range(2, 300+1))
		description="299x299"

	p_threshold=0.05
	
	output_path="dtv_p_%s_bin_predicted_gamma_mc_N_%d_sc_%d_%s.npy"%(str(p_threshold), N_for_fitting, simulations_count, description)

	from scipy.stats import gamma
	
	import numpy as np
	prepare_mean_data=lambda x:np.sqrt(x)
	prepare_variance_data=lambda x:x

	
	x=np.array([N_for_fitting])
	mean_model_x=np.array([prepare_mean_data(x)]).T
	variance_model_x=np.array([prepare_variance_data(x)]).T
	from sklearn import linear_model

	results=dict()
	taker=ns
	if (verbose is True):
		print("Calculating the bins for which the used predicted gamma distribution claims a p closest to the given threshold...")
		from tqdm import tqdm
		taker=tqdm(taker)
	from tqdm import tqdm
	for n in taker:
		mean_for_predicting, var_for_predicting=load_estimated_mean_and_var_for_simulated_distribution(N=N_for_fitting, n=n, simulations_count=simulations_count)
		if (mean_for_predicting is None or var_for_predicting is None):
			print("Missing data for N=%d and n=%d. The calculation will be stopped."%(N_for_fitting, n))
			return
		mean_model=linear_model.LinearRegression()
		mean_model.fit_intercept=fit_intercept
		mean_model.fit(mean_model_x, [mean_for_predicting])
		
		variance_model=linear_model.LinearRegression()
		variance_model.fit_intercept=fit_intercept
		variance_model.fit(variance_model_x, [var_for_predicting])
		
		for N in Ns:
			
			mean=mean_model.predict([prepare_mean_data(np.array([N]))])[0]
			var=variance_model.predict([prepare_variance_data(np.array([N]))])[0]
			
			gamma_alpha=(mean**2)/var
			# the definition of the beta parameter is in Python's SciPy slightly different than the usual mathematical definition
			gamma_beta=var/mean
			gamma_loc=0
			
			# prepare the gamma distribution used to approximate the DTV distribution for the given parameters
			g=gamma(gamma_alpha, loc=gamma_loc, scale=gamma_beta)
			
			# look for the bin with the CDF over the given threshold
			target_bin=None
			i=0
			approximated_value=1
			previous_approximated_value=None
			while(True):
				if (approximated_value<=p_threshold):
					target_bin=i
					# check if the previous approximated p is closer to the desired threshold
					if (previous_approximated_value is not None and abs(p_threshold-previous_approximated_value)<abs(p_threshold-approximated_value)):
						target_bin-=1
					break
				previous_approximated_value=approximated_value
				approximated_value=1-g.cdf(i+CONTINUITY_CORRECTION_FOR_DTV_TO_GAMMA)
				i+=1
			results[(N, n)]=target_bin

	# prepare the results for saving
	import numpy as np
	data=[]
	for N in Ns:
		current=[]
		for n in ns:
			current.append(results[(N, n)])
		data.append(current)
	data=np.array(data)
	# save the obtained results
	np.save(output_path, data)

def test30():
	import numpy as np
	if (False):
		gt_bins=np.load("dtv_p_0.05_bin_exact_199x199.npy", allow_pickle=True)
		gt_gamma_bins=np.load("dtv_p_0.05_bin_gamma_exact_199x199.npy", allow_pickle=True)
		gt_predicted_gamma_bins=np.load("dtv_p_0.05_bin_predicted_gamma_mc_10000000_199x199.npy", allow_pickle=True)
	if (False):
		gt_bins=np.load("dtv_p_0.05_bin_exact_299x299.npy", allow_pickle=True)
		gt_gamma_bins=np.load("dtv_p_0.05_bin_gamma_exact_299x299.npy", allow_pickle=True)
		gt_predicted_gamma_bins=np.load("dtv_p_0.05_bin_predicted_gamma_mc_10000000_299x299.npy", allow_pickle=True)
	if (True):
		gt_bins=np.load("dtv_p_0.05_bin_exact_199x199.npy", allow_pickle=True)
		gt_gamma_bins=np.load("dtv_p_0.05_bin_gamma_exact_199x199.npy", allow_pickle=True)
		gt_predicted_gamma_bins=np.load("dtv_p_0.05_bin_predicted_gamma_mc_N_1000_sc_10000000_199x199.npy", allow_pickle=True)
	if (False):
		gt_bins=np.load("dtv_p_0.05_bin_exact_299x299.npy", allow_pickle=True)
		gt_gamma_bins=np.load("dtv_p_0.05_bin_gamma_exact_299x299.npy", allow_pickle=True)
		gt_predicted_gamma_bins=np.load("dtv_p_0.05_bin_predicted_gamma_mc_N_1000_sc_10000000_299x299.npy", allow_pickle=True)
	take=[]
	take5=[]
	for N in range(gt_bins.shape[0]):
		for n in range(gt_bins.shape[1]):
			take.append((N, n))
			if ((N+1)>=5*(n+1)):
				take5.append((N, n))

	take=take5

	gt_bins=np.array([gt_bins[c] for c in take])
	gt_gamma_bins=np.array([gt_gamma_bins[c] for c in take])
	gt_predicted_gamma_bins=np.array([gt_predicted_gamma_bins[c] for c in take])
	print()
	print("Full match statistics:")
	print()
	print("Ground-truth gamma bins accuracy:           %.2f%%"%(100*np.sum(gt_gamma_bins==gt_bins)/np.prod(gt_bins.shape)))
	print("Ground-truth predicted gamma bins accuracy: %.2f%%"%(100*np.sum(gt_predicted_gamma_bins==gt_bins)/np.prod(gt_bins.shape)))
	print()
	print()
	print()
	print("Matching up to 1 statistics:")
	print()
	print("Ground-truth gamma bins accuracy:           %.5f%%"%(100*np.sum(np.abs(gt_gamma_bins-gt_bins)<=1)/np.prod(gt_bins.shape)))
	print("Ground-truth predicted gamma bins accuracy: %.5f%%"%(100*np.sum(np.abs(gt_predicted_gamma_bins-gt_bins)<=1)/np.prod(gt_bins.shape)))
	#print("Estimated gamma bins accuracy:    %.2f%%"%(100*sum(np.abs(estimated_gamma_bins-gt_bins)<=1)/len(gt_bins)))
	#print("Modelled gamma bins accuracy:     %.2f%%"%(100*sum(np.abs(modelled_gamma_bins-gt_bins)<=1)/len(gt_bins)))
	print()

def test31():
	import numpy as np

	# in the settings here, n is the size of the sampe

	# the larger plot display, but without the focus on the problematic areas
	if (False):
		xticks=[0, 124, 249, 374, 499, 624, 749, 874, 999]
		xtick_labels=[1, 125, 250, 375, 500, 625, 750, 875, 1000]
		yticks=[0, 124, 249, 374, 499, 624, 749, 874, 998]
		ytick_labels=[0.99, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.01]
		input_path="bin_to_norm_1000x999.npy"
	# the smaller plot display with a better focus on the problematic areas
	if (False):
		xticks=[0, 24, 49, 74, 99, 124, 149, 174, 199]
		xtick_labels=[1, 25, 50, 75, 100, 125, 150, 175, 200]
		yticks=[0, 23, 48, 73, 98, 123, 148, 173, 196]
		ytick_labels=[0.99, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.01]
		input_path="bin_to_norm_200x197.npy"
	if (True):
		xticks=[0, 99, 199, 299]
		xtick_labels=[1, 100, 200, 300]
		yticks=[0, 147, 294]
		ytick_labels=[0.99, 0.5, 0.01]
		input_path="bin_to_norm_300x295.npy"

	# the plot settings
	figsize=(21, 8)
	tick_fontsize=40
	label_fontsize=60
	under_fontsize=50
	
	# the image saving settings
	dpi=300
	format="png"

	data=np.load(input_path)
	# this will put n to the x-axis and p to the y-axis
	data=data.T
	# this will assure that the final image will have a commonly used orientation
	data=data[::-1, :]

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	
	fig=plt.figure(figsize=figsize)
	ax1=fig.add_subplot(1, 2, 1)
	plt.ticklabel_format(style="plain")
	plt.imshow(data, cmap="turbo", interpolation="nearest")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.xticks(xticks, xtick_labels)
	plt.yticks(yticks, ytick_labels)
	plt.ylabel("$p$", fontsize=label_fontsize)
	plt.clim(0, 0.05)
	



	if (False):
		xticks=[0, 23, 48, 73, 98, 123, 148, 173, 198]
		xtick_labels=[2, 25, 50, 75, 100, 125, 150, 175, 200]
		yticks=[0, 23, 48, 73, 98, 123, 148, 173, 198]
		ytick_labels=[200, 175, 150, 125, 100, 75, 50, 25, 2]
		input_path="dtv_to_gamma_199x199.npy"
	if (True):
		xticks=[0, 98, 198, 298]
		xtick_labels=[2, 100, 200, 300]
		yticks=[0, 98, 198, 298]
		ytick_labels=[300, 200, 100, 2]
		input_path="dtv_to_gamma_299x299.npy"

	save_path="bin_to_norm_300x295_and_dtv_to_gamma_299x299.png"

	
	
	
	data=np.load(input_path)
	# this will put N to the x-axis and n to the y-axis
	data=data.T
	# this will assure that the final image will have a commonly used orientation
	data=data[::-1, :]
	
	ax2=fig.add_subplot(1, 2, 2)
	plt.ticklabel_format(style="plain")
	plt.imshow(data, cmap="turbo", interpolation="nearest")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.xticks(xticks, xtick_labels)
	plt.yticks(yticks, ytick_labels)
	plt.ylabel("$n$", fontsize=label_fontsize)
	#plt.colorbar()

	for_colorbar=fig.add_axes([0.93, 0.16, 0.03, 0.67])
	plt.colorbar(orientation="vertical", cax=for_colorbar)

	plt.clim(0, 0.05)
	#plt.tight_layout()

	ax1.text(0.5, -0.2, "a)", ha="center", va="center", fontsize=under_fontsize, transform=ax1.transAxes)
	ax2.text(0.5, -0.2, "b)", ha="center", va="center", fontsize=under_fontsize, transform=ax2.transAxes)

	#plt.tight_layout(rect=[0, 0, 1, 1])
	
	
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()
	
def test32():
	with open("pi.txt", "r", encoding="utf8") as f:
		digits=f.read()
	
	save_path="pi.png"
	n=10

	bins=[0]*n
	
	from scipy.stats import chisquare

	chisquare_ps=[]
	dtv_ps=[]

	dtv_Ns=[]
	chisquare_Ns=[]
	for i in range(0, min(len(digits), 500+1)):
		N=i
		d=int(digits[i:i+1])
		bins[d]+=1
	
		if (i<2):
			continue
		_, chisquare_p=chisquare(bins)
		chisquare_ps.append(chisquare_p)
		chisquare_Ns.append(N)

		dtv=get_dtv(bins)
		distribution=load_exact_dtv_distribution(N=N, n=n)
		if (distribution is not None):
			s=0
			for i in range(0, dtv-1+1):
				s+=distribution[i][1]
			dtv_p=1-s/(n**N)
			#print(i, chisquare_p, dtv_p)
			dtv_ps.append(dtv_p)
			dtv_Ns.append(N)
	
	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30
	
	# the image saving settings
	dpi=300
	format="png"

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	plt.ticklabel_format(style="plain")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.ylabel("$p$", fontsize=label_fontsize)
	plt.plot(chisquare_Ns, chisquare_ps, label="chi-square")
	plt.plot(dtv_Ns, dtv_ps, label="CT")
	plt.legend()
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def test33():
	with open("pi.txt", "r", encoding="utf8") as f:
		digits=f.read()
	
	save_path="pi_chisquare.png"
	n=10

	bins=[0]*n
	
	from scipy.stats import chisquare

	chisquare_ps=[]
	
	Ns=[]
	from tqdm import tqdm
	for i in tqdm(range(len(digits))):
		d=int(digits[i:i+1])
		bins[d]+=1
	
		_, chisquare_p=chisquare(bins)
		chisquare_ps.append(chisquare_p)
		Ns.append(i)
	
	# the plot settings
	figsize=(13, 8)
	tick_fontsize=20
	label_fontsize=30
	
	# the image saving settings
	dpi=300
	format="png"

	import matplotlib
	matplotlib.rc("font", size=tick_fontsize)
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=figsize)
	fig.add_subplot(1, 1, 1)
	plt.ticklabel_format(style="plain")
	plt.xlabel("$N$", fontsize=label_fontsize)
	plt.ylabel("$p$", fontsize=label_fontsize)
	plt.plot(Ns, chisquare_ps)
	if (save_path is not None):
		plt.savefig(save_path, format=format, dpi=dpi)
	plt.show()

def main():
	
	# calculate the exact DTV distribution for given values of N and n
	#test1()

	# load a previously calculated exact DTV distribution for given values of N and n
	#test2()

	# generate and cache the results
	#test3()

	# calculate the mean and variance for the cached exact distributions
	#test4()

	# calcualte the accuracy of approximating the binomial distribution with the normal distribution
	#test5()

	# plot the result of test5()
	#test6()

	# calcualte the accuracy of approximating the DTV distribution with the gamma distribution
	#test7()

	# plot the result of test6()
	#test8()
 
 	# fit the mean and the variance of the gamma distribution to a model for given fixed values of N and/or n
	#test9()

	# prepare the coefficients of fitted models for mean and vairance values of the gamma distribution
	#test10()

	# plotting the results of test10()
	#test11()

	# use Monte Carlo to estimate the mean and the variance
	#test12()

	# use Monte Carlo to estimate the mean and the variance for numerous values of N and n and save these estimations
	#test13()

	# compare curve fitting using ground-truth data and Monte Carlo estimation
	#test14()

	# calcualte the accuracy of approximating the DTV distribution with the gamma distribution using the estimated mean and variance values
	#test15()

	# calculate the bins for which the CDF of the gamma distribution with estimated mean and variance is over a given threshold such as 0.95
	#test16()

	# compare the bins for which the ground-truth CDF is over a given threshold such as 0.95 and bins for which the CDF of the gamma distribution with estimated mean and variance is over the same threshold
	#test17()

	# compare the effect of threshold bin calculation based on the gamma distribution parameters obtained through curve fitting
	#test18()

	# similar to test18(), but for multiple target values
	#test19()

	# check the accuracy of obtaining the lower values of DTV
	#test20()

	# plot the histogram of the DTV appearances for given values of N and n
	#test21()

	# plot the histogram of the DTV distribution and the fitted gamma distribution for given values of N and n
	#test22()

	# save the results of statistical tests
	#test23()

	# check the real-life rounding data
	#test24()

	# check the simulated rounding data
	#test25()

	# plot the results of test25()
	#test26()

	# similar to test16(), but for p threshold
	#test27()

	# similar to test27(), but for exact bins
	#test28()

	# similar to test27(), but with using the model to predict the gamma distribution parameters
	#test29()

	# using the results of test27(), teste28(), and test29() for the final statistics
	#test30()

	# combining test6() and test8()
	#test31()

	# check the uniformity of digits of pi
	#test32()

	# check the uniformity of digits of pi
	#test33()

	pass

if (__name__=="__main__"):
	main()
