from __future__ import print_function
import numpy as np
import argparse
import csv
import h5py
import random

np.random.seed(1337)  # for reproducibility

parser = argparse.ArgumentParser()
parser.add_argument("-input", "--input", dest= 'input', type=str, help="meta file including a list of feature files and labels")
parser.add_argument("-f_idx", "--feat_idx", dest= 'feat_idx', type=int, help="feature index (e.g. 8)", default = '8')
parser.add_argument("-base", "--base_dir", dest= 'base_dir', type=str, help="base_dir for feature files")
parser.add_argument("-mt", "--labels", dest= 'labels', type=str, help="labels (idx:idx:..)", default = '3:4:5:6:7')
parser.add_argument("-c_idx", "--c_idx", dest= 'c_idx', type=int, help="cross-validation index (e.g. 3)", default = '2')
parser.add_argument("-w_idx", "--w_idx", dest= 'w_idx', type=int, help="window index for pairwise (e.g. 3) but based on selected labels", default = '3')
parser.add_argument("-n_cc", "--n_cc", dest= 'n_cc', type=int, help="number of cross corpora", default = '0')

parser.add_argument("-out", "--output", dest= 'output', type=str, help="output file in HDF5", default="./output")
parser.add_argument("-f_delim", "--feat_delim", dest= 'feat_delim', type=str, help="feat_delim ", default=";")
parser.add_argument("--nan", help="nan allowed", action="store_true")
parser.add_argument("--delta", help="first delta", action="store_true")
parser.add_argument("--absolute", help="first delta", action="store_true")

args = parser.parse_args()

if args.input == None:
	print('please specify an input meta file')
	exit(1)

meta_file = open(args.input, "r")
f_idx = args.feat_idx

labels = args.labels.split(':')
n_labels = len(labels)

count = -1
input_dim = -1
lines = []

for line in meta_file:
	line = line.rstrip()
	if count == -1:
		count = count + 1
		continue
	params = line.split('\t')
	if input_dim == -1:
		feat_file = params[f_idx]
		if args.base_dir:
			feat_file = args.base_dir + feat_file

		feat_data = np.genfromtxt (feat_file, delimiter=args.feat_delim)
		if len(feat_data.shape) == 1:
			input_dim = feat_data.shape[0]
		else:
			input_dim = feat_data.shape[1]
	
	lines.append(line)
	count = count + 1

n_samples = count

X = np.zeros((n_samples, input_dim * 4))
Y = np.zeros((n_samples, n_labels))

indice_map = {}

print('input dim: ' + str(input_dim))
print('number of samples: '+ str(n_samples))
print('number of labels: '+ str(n_labels))
print('shape', X.shape)
#actual parsing
meta_file.seek(0)
idx = 0

for line in lines:
	line = line.rstrip()
	params = line.split('\t')

	#indice
	cid = int(params[args.c_idx])
	indice = indice_map.get(cid)
	if indice == None:
		indice = [idx]
		indice_map[cid] = indice
	else:
		indice.append(idx)

	print("cid: ", cid)

	#feature file
	feat_file = params[f_idx]

	if args.base_dir:
		feat_file = args.base_dir + feat_file

	feat_data = np.genfromtxt (feat_file, delimiter=args.feat_delim)

	if args.delta:
		n_row = feat_data.shape[0]
		if args.absolute:
			feat_data = np.absolute(feat_data[0:n_row - 1] - feat_data[1:n_row])
		else:
			feat_data = feat_data[0:n_row - 1] - feat_data[1:n_row]

	#print(feat_data)
	if args.nan:
		X[idx, 0:input_dim] = np.nanmean(feat_data, axis = 0)
		X[idx, input_dim:input_dim * 2] = np.nanmedian(feat_data, axis = 0)
		X[idx, input_dim * 2:input_dim * 3] = np.nanmax(feat_data, axis = 0)
		X[idx, input_dim * 3:input_dim * 4] = np.nanstd(feat_data, axis = 0)
	else:
		X[idx, 0:input_dim] = np.mean(feat_data, axis = 0)
		X[idx, input_dim:input_dim * 2] = np.median(feat_data, axis = 0)
		X[idx, input_dim * 2:input_dim * 3] = np.max(feat_data, axis = 0)
		X[idx, input_dim * 3:input_dim * 4] = np.std(feat_data, axis = 0)
	#print(X[idx,:])

	#copy labels
	for lab_idx in range(n_labels):
		Y[idx, lab_idx] = params[int(labels[lab_idx])]
	
	idx = idx + 1
	print("processing: ", idx, " :", feat_file)


print('successfully write samples: ' + str(idx))

h5_output = args.output + '.h5'

if args.n_cc > 0:
	idx = 0
	# loading and constructing each fold in memory takes too much time.
	index_list = []
	start_indice = np.zeros((args.n_cc))
	end_indice = np.zeros((args.n_cc))

	X_ordered = np.zeros((n_samples, input_dim * 4))
	Y_ordered = np.zeros((n_samples, n_labels))

	start_idx = 0
	end_idx = 0

	accumulated_n_samples = 0
	for cid, indice in indice_map.items():
		#print('indice', indice)
		if indice == None:
			continue
		X_temp = X[indice]
		Y_temp = Y[indice]
		Y_temp[:,args.w_idx] += accumulated_n_samples

		end_idx = start_idx + X_temp.shape[0]
		print('shape', X_temp.shape)
		start_indice[idx] = start_idx
		end_indice[idx] = end_idx
		print("corpus: ", idx, " starting from: ", start_idx, " ends: ", end_idx)
		X_ordered[start_idx:end_idx] = X_temp
		Y_ordered[start_idx:end_idx] = Y_temp
		print("starting window idx: ", Y_temp[0, args.w_idx])
		start_idx = end_idx
		idx = idx + 1

		accumulated_n_samples += len(indice)
		
	print("shape of feat: ", X_ordered.shape)
	print("shape of label: ", Y_ordered.shape)
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('feat', data= X_ordered)
		hf.create_dataset('label', data= Y_ordered)
		hf.create_dataset('start_indice', data=start_indice)
		hf.create_dataset('end_indice', data=end_indice)
	print('total cv: ' + str(len(start_indice)))	
	'''
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('feat', data=X)
		hf.create_dataset('label', data=Y)'''
else:
	print("shape of feat: ", X.shape)
	print("shape of label: ", Y.shape)
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('feat', data=X)
		hf.create_dataset('label', data=Y)



meta_file.close()
