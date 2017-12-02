from __future__ import print_function
import numpy as np
import argparse
import csv
import h5py
import random
import feature_utility as fu
import os, sys
import cv2
np.random.seed(1337)  # for reproducibility

parser = argparse.ArgumentParser()
parser.add_argument("-input", "--input", dest= 'input', type=str, help="meta file including a list of feature files and labels")
parser.add_argument("-mt", "--multitasks", dest= 'multitasks', type=str, help="multi-tasks (idx:idx:..)", default = '3:4:5:6:7')
parser.add_argument("-pose_idx", "--pose_feat_idx", dest= 'pose_feat_idx', type=int, help="feature index (e.g. 8)", default = '8')
parser.add_argument("-emo_idx", "--emo_idx", dest= 'emo_idx', type=int, help="image folder index (e.g. 3)", default = '9')
parser.add_argument("-c_idx", "--c_idx", dest= 'c_idx', type=int, help="cross-validation index (e.g. 3)", default = '3')
parser.add_argument("-n_cc", "--n_cc", dest= 'n_cc', type=int, help="number of cross corpora", default = '0')

parser.add_argument("-pose_m_steps", "--pose_max_time_steps", dest= 'pose_max_time_steps', type=int, help="maximum time steps", default = '50')
parser.add_argument("-emo_m_steps", "--emo_max_time_steps", dest= 'emo_max_time_steps', type=int, help="maximum time steps", default = '50')
parser.add_argument("-pose_c_len", "--pose_context_length", dest= 'pose_context_length', type=int, help="context window length", default = '1')
parser.add_argument("-emo_c_len", "--emo_context_length", dest= 'emo_context_length', type=int, help="context window length", default = '1')

parser.add_argument("-out", "--output", dest= 'output', type=str, help="output file in HDF5", default="./output")
parser.add_argument("-f_delim", "--feat_delim", dest= 'feat_delim', type=str, help="feat_delim ", default=";")
#parser.add_argument("-img_base_dir", "--img_base_dir", dest= 'img_base_dir', type=str, help="img_base_dir ", default="./")


parser.add_argument("--two_d", help="two_d",
                    action="store_true")
parser.add_argument("--three_d", help="three_d",
                    action="store_true")
parser.add_argument("--headerless", help="headerless in feature file?",
                    action="store_true")
parser.add_argument("--evaluation", help="evaluation",
                    action="store_true")
parser.add_argument("--no_image", help="no_image",
                    action="store_true")


args = parser.parse_args()

if args.input == None:
	print('please specify an input meta file')
	exit(1)

meta_file = open(args.input, "r")
pose_idx = args.pose_feat_idx
n_cc = args.n_cc

pose_max_t_steps = args.pose_max_time_steps
emo_max_t_steps = args.emo_max_time_steps

pose_input_dim = -1
emo_input_dim = -1
feat_delim = args.feat_delim
pose_context_length = args.pose_context_length
emo_context_length = args.emo_context_length


#parsing 
count = -1
lines = []

for line in meta_file:
	line = line.rstrip()
	if count == -1:
		count = count + 1
		continue
	params = line.split('\t')
	if pose_input_dim == -1:
		feat_file = params[pose_idx]
		feat_data = np.genfromtxt (feat_file, delimiter=feat_delim)
		if len(feat_data.shape) == 1:
			pose_input_dim = 1
		else:
			pose_input_dim = feat_data.shape[1]
	
	if emo_input_dim == -1:
		feat_file = params[emo_idx]
		feat_data = np.genfromtxt (feat_file, delimiter=feat_delim)
		if len(feat_data.shape) == 1:
			emo_input_dim = 1
		else:
			emo_input_dim = feat_data.shape[1]

	lines.append(line)
	count = count + 1

#randomise	
if n_cc == 0:
	random.shuffle(lines)

n_samples = count

labels = args.multitasks.split(':')
n_labels = len(labels)

pose_max_t_steps = int(pose_max_t_steps / pose_context_length)

if args.two_d:
	X_pose = np.zeros((n_samples, pose_max_t_steps, 1, pose_context_length, pose_input_dim))
elif args.three_d:
	X_pose = np.zeros((n_samples, 1, pose_max_t_steps, pose_context_length, pose_input_dim))
else:
	X_pose = np.zeros((n_samples, pose_max_t_steps, pose_input_dim * pose_context_length))

emo_max_t_steps = int(emo_max_t_steps / emo_context_length)
if args.two_d:
	X_emo = np.zeros((n_samples, emo_max_t_steps, 1, emo_context_length, emo_input_dim))
elif args.three_d:
	X_emo = np.zeros((n_samples, 1, emo_max_t_steps, emo_context_length, emo_input_dim))
else:
	X_emo = np.zeros((n_samples, emo_max_t_steps, emo_input_dim * emo_context_length))

Y = np.zeros((n_samples, n_labels))

indice_map = {}

print('input dim: ' , input_dim)
print('number of samples: ', n_samples)
print('number of labels: ', n_labels)
print('pose_max steps: ', pose_max_t_steps)
print('pose_context windows: ', pose_context_length)
print('emo_max steps: ', emo_max_t_steps)
print('emo_context windows: ', emo_context_length)
print('pose shape', X_pose.shape)
print('emo shape', X_emo.shape)

#actual parsing
meta_file.seek(0)
idx = 0

for line in lines:
	line = line.rstrip()
	params = line.split('\t')

	cid = int(params[args.c_idx])

	indice = indice_map.get(cid)
	if indice == None:
		indice = [idx]
		indice_map[cid] = indice
	else:
		indice.append(idx)

	#feature file for pose
	feat_file = params[pose_idx]
	feat_data = np.genfromtxt (feat_file, delimiter=feat_delim)

	#2d with context windows
	if args.two_d:
		for t_steps in range(pose_max_t_steps):
			if t_steps * pose_context_length < feat_data.shape[0] - pose_context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(pose_context_length):
					X_pose[idx, t_steps, 0, c, ] = feat_data[t_steps * pose_context_length + c]
	elif args.three_d:#3d with context windows
		for t_steps in range(pose_max_t_steps):
			if t_steps * pose_context_length < feat_data.shape[0] - pose_context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(pose_context_length):
					X_pose[idx, 0, t_steps, c, ] = feat_data[t_steps * pose_context_length + c]
	##1d but context windows copy features into time slots
	elif pose_context_length == 1:
		for t_steps in range(pose_max_t_steps):
			if t_steps < feat_data.shape[0]:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				X_pose[idx, t_steps,] = feat_data[t_steps]
	else:#1d but context windows
		for t_steps in range(pose_max_t_steps):
			if t_steps * pose_context_length < feat_data.shape[0] - pose_context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(pose_context_length):
					X_pose[idx, t_steps, c * input_dim: (c + 1) * input_dim] = feat_data[t_steps * context_length + c]
	#EMO
	#feature file for pose
	feat_file = params[emo_idx]
	feat_data = np.genfromtxt (feat_file, delimiter=feat_delim)
	if args.two_d:
		for t_steps in range(emo_max_t_steps):
			if t_steps * emo_context_length < feat_data.shape[0] - emo_context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(emo_context_length):
					X_emo[idx, t_steps, 0, c, ] = feat_data[t_steps * emo_context_length + c]
	elif args.three_d:#3d with context windows
		for t_steps in range(emo_max_t_steps):
			if t_steps * emo_context_length < feat_data.shape[0] - emo_context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(emo_context_length):
					X_emo[idx, 0, t_steps, c, ] = feat_data[t_steps * emo_context_length + c]
	##1d but context windows copy features into time slots
	elif emo_context_length == 1:
		for t_steps in range(emo_max_t_steps):
			if t_steps < feat_data.shape[0]:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				X_emo[idx, t_steps,] = feat_data[t_steps]
	else:#1d but context windows
		for t_steps in range(emo_max_t_steps):
			if t_steps * emo_context_length < feat_data.shape[0] - emo_context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(emo_context_length):
					X_emo[idx, t_steps, c * input_dim: (c + 1) * input_dim] = feat_data[t_steps * context_length + c]
	
	
	#copy labels
	for lab_idx in range(n_labels):
		Y[idx, lab_idx] = params[int(labels[lab_idx])]

	idx = idx + 1
	print("processing: ", idx, " :", feat_file)
print('successfully write samples: ' + str(idx))

h5_output = args.output + '.h5'


if n_cc > 0:
	idx = 0
	# loading and constructing each fold in memory takes too much time.
	index_list = []
	start_indice = np.zeros((n_cc))
	end_indice = np.zeros((n_cc))

	if args.two_d:
		X_pose_ordered = np.zeros((n_samples, pose_max_t_steps, 1, pose_context_length, pose_input_dim))
	elif args.three_d:
		X_pose_ordered = np.zeros((n_samples, 1, pose_max_t_steps, pose_context_length, pose_input_dim))
	else:
		X_pose_ordered = np.zeros((n_samples, pose_max_t_steps, pose_input_dim * pose_context_length))
	
	if args.two_d:
		X_emo_ordered = np.zeros((n_samples, emo_max_t_steps, 1, emo_context_length, emo_input_dim))
	elif args.three_d:
		X_emo_ordered = np.zeros((n_samples, 1, emo_max_t_steps, emo_context_length, emo_input_dim))
	else:
		X_emo_ordered = np.zeros((n_samples, emo_max_t_steps, emo_input_dim * emo_context_length))
	

	Y_ordered = np.zeros((n_samples, n_labels))

	start_idx = 0
	end_idx = 0
	for cid, indice in indice_map.items():
		#print('indice', indice)
		if indice == None:
			continue
		X_pose_temp = X_pose[indice]
		Y_temp = Y[indice]
		end_idx = start_idx + X_pose_temp.shape[0]
		print('shape', X_pose_temp.shape)
		start_indice[idx] = start_idx
		end_indice[idx] = end_idx
		print("corpus: ", idx, " starting from: ", start_idx, " ends: ", end_idx)
		X_pose_ordered[start_idx:end_idx] = X_pose_temp
		X_emo_ordered[start_idx:end_idx] = X_emo[indice]

		Y_ordered[start_idx:end_idx] = Y_temp
		start_idx = end_idx
		idx = idx + 1
		
	print("shape of pose feat(v_feat): ", X_pose_ordered.shape)
	print("shape of emo feat(a_feat): ", X_emo_ordered.shape)
	print("shape of label: ", Y_ordered.shape)
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('v_feat', data= X_pose_ordered)
		hf.create_dataset('a_feat', data= X_emo_ordered)
		hf.create_dataset('label', data= Y_ordered)
		hf.create_dataset('start_indice', data=start_indice)
		hf.create_dataset('end_indice', data=end_indice)
	print('total cv: ' + str(len(start_indice)))	
	
else:
	print("shape of v_feat: ", X_pose.shape)
	print("shape of a_feat: ", X_emo.shape)
	print("shape of label: ", Y.shape)
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('v_feat', data=X_pose)
		hf.create_dataset('a_feat', data=X_emo)
		hf.create_dataset('label', data=Y)

meta_file.close()
