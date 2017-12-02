from __future__ import print_function
import numpy as np
np.random.seed(1337)
from sklearn.metrics import f1_score,recall_score,confusion_matrix
from sklearn import svm, linear_model, cross_validation
import scipy.stats as st
import itertools
import h5py
import sys
from keras import regularizers
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from helper import *

class ConvDNN():

	def __init__(self, args):
		self.build_model()

	def fit(self, X_train, y_train, X_valid, y_valid, batch_size = 128, epochs = 20, callbacks = None):
		
		X_train, y_train = transform_pairwise_multi_data(X_train, y_train)
		X_valid, y_valid = transform_pairwise_multi_data(X_valid, y_valid)

		Y_train_valid = one_hot_vector(self.num_classes, y_train, y_valid)
		
		self.merged_discriminator.fit(X_train, Y_train_valid[0], batch_size = batch_size, nb_epoch=epochs, validation_data=(X_valid, Y_train_valid[1]), callbacks=callbacks)

	def score(self, X, y):
		"""
		Because we transformed into a pairwise problem, chance level is at 0.5
		"""
		X_trans, y_trans = transform_pairwise_multi_data(X, y)

		predictions = self.merged_discriminator.predict(X_trans)

		labels = np.argmax(y_trans,1)
		pred = np.argmax(predictions[count],1)

		#tau distance
		tau_d =  1. - np.mean(pred == labels)
		#tau correlation
		tau, p_value = st.kendalltau(pred, labels)
		#confusion matrix
		cm = confusion_matrix(labels, pred)
		prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		X_errors = X_trans[pred != labels]
		Y_errors = labels[pred != labels]
		X_Y_errors = np.c_[ X_errors, Y_errors ]

		X_correct = X_trans[pred == labels]
		Y_correct = labels[pred == labels]
		X_Y_correct = np.c_[ X_correct, Y_correct ]

		return tau_d, tau, cm, prob_cm, X_Y_errors, X_Y_correct
	def parsing_CNN_configuration(args):
		
		modalities = []
		for modality in args.modality.split(";"):
			modalities.append(modality)
			
		n_row = str_to_int(args.r_nrow.split(";"))
		n_col = str_to_int(args.r_ncol.split(";"))
		
		if args.input_dim == 3:
			n_time = str_to_int(args.r_ntime.split(";"))
		else:
			n_time = [int(x) for x in range(len(modalities))]
		
		depth = str_to_int(args.depth.split(";"))

		n_kernels = []
		for mv in args.n_kernels.split(";"):
			n_kernels.append(str_to_int(mv.split(",")))

		#parsing resolutions of convolutional layers
		
		crows = []
		ccols = []
		for mv in args.cnn_n_row.split(";"):		
			crows.append(str_to_int(mv.split(",")))
		for mv in args.cnn_n_col.split(";"):		
			ccols.append(str_to_int(mv.split(",")))
		
		if args.input_dim == 3:
			ctimes = []
			for mv in args.cnn_n_time.split(";"):		
				ctimes.append(str_to_int(mv.split(",")))
				
		kernel_resolution = []
		for i in range(len(modalities)):
			if len(crows[i]) == len(ccols[i]) and len(ccols[i]) == depth[i]:
				
				if args.input_dim == 3 and len(ctimes[i]) != depth[i]:
					print("check structure of CNN D!")
					raise ValueError("parsing argument error!")
					
				kernel_r_c = []
				for d in range(len(crows[i])):
					if args.input_dim == 3:
						kernel_r_c.append((ctimes[i][d], crows[i][d], ccols[i][d]))
					else:
						kernel_r_c.append((crows[i][d], ccols[i][d]))
				kernel_resolution.append( kernel_r_c )
			else:
				print("check structure of CNN D!")
				raise ValueError("parsing argument error!")
		
		#parsing pooling layers
		pooling_resolution = []
		
		if args.pool_n_row and args.pool_n_col:
			prows = []
			pcols = []
			for mv in args.pool_n_row.split(";"):		
				prows.append(str_to_int(mv.split(",")))
			for mv in args.pool_n_col.split(";"):		
				pcols.append(str_to_int(mv.split(",")))
			
			if args.input_dim == 3:
				ptimes = []
				for mv in args.pool_n_time.split(";"):		
					ptimes.append(str_to_int(mv.split(",")))
				
			for i in range(len(modalities)):
				if len(crows[i]) == len(ccols[i]) and len(ccols[i]) == len(prows[i]) and len(prows[i]) == len(pcols[i]) and len(pcols[i]) == depth[i]:
					if args.input_dim == 3 and len(ptimes[i]) != depth[i]:
						print("check structure of pooling of CNN D!")
						raise ValueError("parsing argument error!")
					kernel_r_c = []
					for d in range(len(prows[i])):
						if args.input_dim == 3:
							kernel_r_c.append((ptimes[i][d], prows[i][d], pcols[i][d]))
						else:
							kernel_r_c.append((prows[i][d], pcols[i][d]))
					pooling_resolution.append( kernel_r_c )
				else:
					print("crows ccols ccols prows prows pcols pcols depth of D: ", len(crows[i]), len(ccols[i]),len(ccols[i]), len(prows[i]),len(prows[i]), len(pcols[i]),len(pcols[i]), depth[i])
					print("check structure of pooling of CNN D!")
					raise ValueError("parsing argument error!")
		
		if args.D_weight_regulariser:
			regulariser = regularizers.l2(args.D_weight_regulariser)
		else:
			regulariser = None
			
		return modalities, n_time, n_row, n_col, depth, n_kernels, kernel_resolution, pooling_resolution, regulariser


	def build_model(self):

		discriminators = []
	
		#parsing configurations
		modalities, n_time, n_row, n_col, depth, n_kernels, kernel_resolution, pooling_resolution, regulariser = self.parsing_CNN_configuration(args)

		self.num_classes = n_class
		self.earlystopping = []
		self.modality = modalities
	
		optimizer = Adam(g_lr, 0.5)
		loss = 'categorical_crossentropy'

		for i in range(len(modalities)):
			
			#build a discriminator
			img, D = self.building_discriminator(depth[i], n_kernels[i], kernel_resolution[i], pooling_resolution[i], n_time[i], n_row[i], n_col[i], args.dropout, modalities[i], regulariser)
			
			D = Flatten()(D)
			inputs.append(img)
			discriminators.append(D)
		
		#merging discriminators
		if len(modalities) > 1:
			next_merged = Concatenate(name="concate")(discriminators)
		else:
			next_merged = discriminators[0]
				
		#fuly connected layers for merged(or single) D
		for i in range(args.depth_merged):
			next_merged = Dense(args.n_node, activation="relu")(next_merged)
	   
		label = Dense(n_class, activation="softmax")(next_merged)
		self.merged_discriminator = Model(inputs, label)

		self.merged_discriminator.compile(loss = loss, 
			optimizer=optimizer,
			metrics=['accuracy'])
		
		print("Current configuration for merged_discriminator")
		self.merged_discriminator.summary()

	def building_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_time, n_row, n_col, dropout, modalities, regulariser):
		if args.input_dim == 3:
			img, D = self.building_3D_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_time, n_row, n_col, dropout, modalities, regulariser)
		elif args.input_dim == 2:
			img, D = self.building_2D_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_row, n_col, dropout, modalities, regulariser)
		else:
		   	img, D = self.building_1D_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_row, dropout, modalities, regulariser)
		return img, D

	def building_1D_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_row, dropout, prefix_name = "audio", regulariser = None):
	
		layer_idx = 0
		
		img_shape = (n_row)
		img = Input(shape=img_shape, name = prefix_name + "_input_" + str(layer_idx) )
		layer_idx +=1
		
		#first layer
		next = Conv1D(n_kernels[0], kernel_size=kernel_resolution[0], strides=2, input_shape=img_shape, padding="same",name = prefix_name + "_cnn_" + str(layer_idx), kernel_regularizer = regulariser )(img)
		layer_idx +=1
		next = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next)
		layer_idx +=1
		if dropout:
			next = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next)
			layer_idx +=1
			
		next = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next)
		layer_idx +=1
		
		if len(pooling_resolution) > 0:
			next = MaxPooling1D(pool_size = pooling_resolution[0],name = prefix_name + "_maxpooling_" + str(layer_idx))(next)
			layer_idx +=1
		
		#middle layers
		for idx in range(depth):
			next = Conv1D(n_kernels[idx], kernel_size=kernel_resolution[idx], padding="same",name = prefix_name + "_cnn_" + str(layer_idx), kernel_regularizer = regulariser)(next)
			layer_idx +=1
			next = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next)
			layer_idx +=1
			
			if dropout:
				next = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next)
				layer_idx +=1
			
			next = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next)
			layer_idx +=1
		
			if len(pooling_resolution) > 0:
				next = MaxPooling1D(pool_size = pooling_resolution[idx],name = prefix_name + "_maxpooling_" + str(layer_idx))(next)

		return img, next

	def building_2D_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_row, n_col, dropout, prefix_name = "audio", regulariser = None):
	
		layer_idx = 0
		
		img_shape = (n_row, n_col, 1)
		img = Input(shape=img_shape, name = prefix_name + "_input_" + str(layer_idx) )
		layer_idx +=1
		
		#first layer
		next = Conv2D(n_kernels[0], kernel_size=kernel_resolution[0], strides=2, input_shape=img_shape, padding="same",name = prefix_name + "_cnn_" + str(layer_idx), kernel_regularizer = regulariser )(img)
		layer_idx +=1
		next = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next)
		layer_idx +=1
		if dropout:
			next = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next)
			layer_idx +=1
			
		next = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next)
		layer_idx +=1
		
		if len(pooling_resolution) > 0:
			next = MaxPooling2D(pool_size = pooling_resolution[0],name = prefix_name + "_maxpooling_" + str(layer_idx))(next)
			layer_idx +=1
		
		#middle layers
		for idx in range(depth):
			next = Conv2D(n_kernels[idx], kernel_size=kernel_resolution[idx], padding="same",name = prefix_name + "_cnn_" + str(layer_idx), kernel_regularizer = regulariser)(next)
			layer_idx +=1
			next = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next)
			layer_idx +=1
			
			if dropout:
				next = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next)
				layer_idx +=1
			
			next = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next)
			layer_idx +=1
		
			if len(pooling_resolution) > 0:
				next = MaxPooling2D(pool_size = pooling_resolution[idx],name = prefix_name + "_maxpooling_" + str(layer_idx))(next)

		return img, next

	def building_3D_discriminator(depth, n_kernels, kernel_resolution, pooling_resolution, n_time, n_row, n_col, dropout, prefix_name = "audio", regulariser = None):
	
		layer_idx = 0
		
		img_shape = (n_time, n_row, n_col, 1)
		img = Input(shape=img_shape, name = prefix_name + "_input_" + str(layer_idx) )
		layer_idx +=1
		
		#first layer
		next = Conv3D(n_kernels[0], kernel_size=kernel_resolution[0], strides=2, input_shape=img_shape, padding="same",name = prefix_name + "_cnn_" + str(layer_idx), kernel_regularizer = regulariser )(img)
		layer_idx +=1
		next = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next)
		layer_idx +=1
		if dropout:
			next = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next)
			layer_idx +=1
			
		next = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next)
		layer_idx +=1
		
		if len(pooling_resolution) > 0:
			next = MaxPooling3D(pool_size = pooling_resolution[0],name = prefix_name + "_maxpooling_" + str(layer_idx))(next)
			layer_idx +=1
		
		#middle layers
		for idx in range(depth):
			next = Conv3D(n_kernels[idx], kernel_size=kernel_resolution[idx], padding="same",name = prefix_name + "_cnn_" + str(layer_idx) , kernel_regularizer = regulariser)(next)
			layer_idx +=1
			next = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next)
			layer_idx +=1
			
			if dropout:
				next = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next)
				layer_idx +=1
			
			next = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next)
			layer_idx +=1
		
			if len(pooling_resolution) > 0:
				next = MaxPooling3D(pool_size = pooling_resolution[idx],name = prefix_name + "_maxpooling_" + str(layer_idx))(next)

		return img, next