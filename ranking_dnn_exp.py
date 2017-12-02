
import argparse
import numpy as np
np.random.seed(1337) 
import sys
import h5py
import os
from cnndnn import *
from helper import *



if __name__ == '__main__':
	# as showcase, we will create some non-linear data
	# and print the performance of ranking vs linear regression
	parser = argparse.ArgumentParser()
	
	#cv 
	parser.add_argument("-dt", "--data", dest= 'data', type=str, help="path of DB")
	parser.add_argument("-test_idx", "--test_idx", dest= 'test_idx', type=str, help="indice of test sets, separated by commas")
	parser.add_argument("-train_idx", "--train_idx", dest= 'train_idx', type=str, help="indice of train sets, separated by commas")
	parser.add_argument("-valid_idx", "--valid_idx", dest= 'valid_idx', type=str, help="indice of validation sets, separated by commas")
	parser.add_argument("-ignore_idx", "--ignore_idx", dest= 'ignore_idx', type=str, help="(0,1,2,3,4)")

	#setup
	parser.add_argument("-b", "--batch", dest= 'batch', type=int, help="batch size", default=128)
	parser.add_argument("-e", "--epoch", dest= 'epoch', type=int, help="maximum number of epoch", default=10)
	
	#feat
	parser.add_argument("-w_id", "--window_idx", dest= 'window_idx', type=int, help="index of window in labels", default=3)
	parser.add_argument("-c_id", "--class_idx", dest= 'class_idx', type=int, help="index of class in labels", default=4)
	parser.add_argument("-max_feat", "--max_feat", dest= 'max_feat', type=int, help="max number of features for selection", default=50)
	
	#shape
	parser.add_argument("-mod", "--modality", dest= 'modality', type=str, help="input feature name in hd5 DB", default='feat')
	parser.add_argument("-r_ntime", "--r_ntime", dest= 'r_ntime', type=str, help="reshaped # times, for multimodal features, separated by ;", default = "10;1")
	parser.add_argument("-r_nrow", "--r_nrow", dest= 'r_nrow', type=str, help="reshaped # rows, for multimodal features, separated by ;", default = "28;48")
	parser.add_argument("-r_ncol", "--r_ncol", dest= 'r_ncol', type=str, help="reshaped # columns, for multimodal features, separated by ;", default = "28;48")

	
	parser.add_argument("-lr", "--learningrate", dest= 'lr', type=float, help="learning rate for a discriminator", default=0.0002)
	parser.add_argument("-reg", "--weight_regulariser", dest= 'weight_regulariser', type=float, help="weight regulariser for discriminator(s)")
	
	#convolution
	parser.add_argument("-depth", "--depth", dest = 'depth', type = str, help = "depth of discriminator(s)", default = "4;4")
	
	parser.add_argument("-depth_merged", "--depth_merged", dest = 'depth_merged', type = int, help = "depth of fully connected layer for a combined discriminator", default = 1)

	parser.add_argument("-n_node", "--n_node", dest = 'n_node', type = int, help = "# nodes of fully-connected layers after CNNs", default = 512)
	parser.add_argument("-drop","--dropout", dest='dropout', type=float, help="dropout")
	
	parser.add_argument("-n_kernels","--n_kernels", dest='n_kernels', type=str, help="# kernels in discriminator(s)", default = "16,32,64,128;16,32,64,128")
	parser.add_argument("-cnn_n_time","--cnn_n_time", dest='cnn_n_time', type=str, help="# times in CNN for discriminator(s)", default = "4,4,4,4;4,4,4,4")
	parser.add_argument("-cnn_n_row","--cnn_n_row", dest='cnn_n_row', type=str, help="# rows in CNN for discriminator(s)", default = "4,4,4,4;4,4,4,4")
	parser.add_argument("-cnn_n_col","--cnn_n_col", dest='cnn_n_col', type=str, help="# cols in CNN for discriminator(s)", default = "4,4,4,4;4,4,4,4")
	
	parser.add_argument("-pool_n_time","--pool_n_time", dest='pool_n_time', type=str, help="# times in pooling layers for discriminator(s)", default = "2,2,2,2;2,2,2,2")
	parser.add_argument("-pool_n_row","--pool_n_row", dest='pool_n_row', type=str, help="# rows in pooling layers for discriminator(s)", default = "2,2,2,2;2,2,2,2")
	parser.add_argument("-pool_n_col","--pool_n_col", dest='pool_n_col', type=str, help="# cols in pooling layers for discriminator(s)", default = "2,2,2,2;2,2,2,2")
	
	#etc
	parser.add_argument("-log", "--log_file", dest= 'log_file', type=str, help="log file path")
	parser.add_argument("-error", "--error_file", dest= 'error_file', type=str, help="error file path")

	parser.add_argument("--feat_select", help="feature section", action="store_true")
	parser.add_argument("--pairwise", help="pairwise feature section (note very slow!)", action="store_true")
	parser.add_argument("--feature_analysis", help="pairwise feature analysis using the whole data", action="store_true")
	parser.add_argument("--spearmanr", help="spearmanr instead of kendall tau", action="store_true")
	
	args = parser.parse_args()

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	
	if args.log_file:
		log_writer = open(args.log_file, 'a')
	else:
		log_writer = None

	if args.error_file:
		error_writer = open(args.error_file + ".error.csv", 'a')
		correct_writer = open(args.error_file + ".correct.csv", 'a')
	else:
		error_writer = None
		correct_writer = None

	#write options
	log(str(args), log_writer)
	
	train_idx, test_idx, valid_idx, ignore_idx, adopt_idx, kf_idx = compose_idx(args.train_idx, args.test_idx, args.valid_idx, args.ignore_idx, None, None)
	
	with h5py.File(args.data,'r') as hf:
		print('List of arrays in this file: \n', hf.keys())
		train_csv = []
		if args.modality:
			for modality in args.modality.split(";"):
				data = hf.get(modality)
				s_train_csv = np.array(data)

				#correct errors
				s_train_csv[np.isnan(s_train_csv) + np.isinf(s_train_csv)] = 0.
				print('Shape of the array %s feat: %s'% (modality, str(s_train_csv.shape)))
				train_csv.append(s_train_csv)

		#labels
		data = hf.get('label')
		train_lab = np.array(data)
				
		print('Shape of the array lab: ', train_lab.shape)
		if len(test_idx) > 0:
			start_indice = np.array(hf.get('start_indice'))
			end_indice = np.array(hf.get('end_indice'))
			print('Shape of the indice for start: ', start_indice.shape)

	n_modality = len(args.modality.split(";"))
	if args.feature_analysis:
		for idx in range(n_modality):
			feature_analysis(train_csv[idx], train_lab, args, log_writer)	   
		log_writer.close()
	else:

		if len(test_idx) > 0 :

			#compose cross validation sets
			X_train, X_test, X_valid, Y_train, Y_test, Y_valid = compose_multi_data_set(test_idx, valid_idx, train_idx, train_csv, train_lab, start_indice, end_indice)
			
			#choose labels for class and window
			Y_train = Y_train[:, (args.class_idx, args.window_idx)]
			Y_test = Y_test[:, (args.class_idx, args.window_idx)]
			Y_valid = Y_valid[:, (args.class_idx, args.window_idx)]
			Y_train = Y_train.astype(int)
			Y_test = Y_test.astype(int)
			Y_valid = Y_valid.astype(int)

			#feature selection using validation data
			if args.feat_select:
				for idx in range(n_modality):
					fs_X_train, fs_X_test, fs_X_valid = feature_selection(X_train[idx], X_test[idx], X_valid[idx], Y_train[idx], Y_test[idx], Y_valid[idx], args, log_writer)
					X_train[idx] = fs_X_train
					X_test[idx] = fs_X_test
					X_valid[idx] = fs_X_valid

			#bruth force search for best params
			best_model = None
			best_tau_d = 1.0
			best_p_cm = None
			best_param = 0.0
			for c in args.complexity.split(","):
								
				rank_model = ConvDNN(args)
				rank_model.fit(X_train, Y_train, X_valid, Y_valid, args.batch, args.epoch)

				tau_d, tau, cm, prob_cm, X_Y_errors, X_Y_correct = rank_model.score(X_train, Y_train)
				log('Complexity %.3f, performance of ranking on training data: tau_d: %.3f, prob_cm: %s' %(c, tau_d, str(prob_cm)), log_writer)

				tau_d, tau, cm, prob_cm, X_Y_errors, X_Y_correct = rank_model.score(X_valid, Y_valid)
				log('Complexity %.3f, performance of ranking on validataion data: tau_d: %.3f, prob_cm: %s' %(c, tau_d, str(prob_cm)), log_writer)

				#tau_d is error rate
				if tau_d < best_tau_d:
					best_model = rank_model
					best_tau_d = tau_d
					best_param = c
					best_p_cm = prob_cm

			log('validation: best complexity: %.3f and its performance: %.3f, prob_cm: %s' %(best_param, best_tau_d, str(best_p_cm)), log_writer)
			
			tau_d, tau, cm, prob_cm, X_Y_errors, X_Y_correct = best_model.score(X_test, Y_test)
			
			log('performance of ranking on testing data: tau_d: %.3f, prob_cm: %s' %(tau_d, str(prob_cm)), log_writer)

			#writing errors and correct samples
			if error_writer and correct_writer:
				log("errors were written into a file", log_writer)
				np.savetxt(error_writer, X_Y_errors, fmt='%.4e')

				log("correct samples were written into a file", log_writer)
				np.savetxt(correct_writer, X_Y_correct, fmt='%.4e')

				correct_writer.close()
				error_writer.close()

		log_writer.close()
			

	