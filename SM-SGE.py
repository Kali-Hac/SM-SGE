import time
import numpy as np
import tensorflow as tf
import os, sys
from models import MGRN_S
from utils import process_i as process
from tensorflow.python.layers.core import Dense
from sklearn.preprocessing import label_binarize


dataset = 'IAS'
split = 'B'
pretext = 'recon'
RN_dir = 'RN/'  # save MLP models for person Re-ID
pre_dir = 'Pre-Trained/'    # save self-supervised SM-SGE models
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
pre_epochs = 200    # epochs for self-supervised training
sample_num = 1      # number of sampling rounds (r)
nb_nodes = 20       # number of nodes in joint-scale graph
nhood = 1           # structural relation learning (nhood=1 for neighbor nodes)
fusion_lambda = 1   # collaboration fusion coefficient
ft_size = 3         # originial node feature dimension (D)
time_step = 6       # sequence length (f)

LSTM_embed = 256    # number of hidden units per layer in LSTM (D_{h})
num_layers = 2      # number of LSTM layers

# training params
batch_size = 256
nb_epochs = 100000
patience = 150     # patience for early stopping
lr = 0.005  # learning rate
hid_units = [8]  # numbers of hidden units per each attention head in each layer
Ps = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = MGRN_S

tf.app.flags.DEFINE_string('dataset', 'BIWI', "Dataset: BIWI, IAS, KS20 or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")
tf.app.flags.DEFINE_string('split', '', "for IAS-Lab testing splits")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('task', 'pre', "prediction")     # we regard reconstruction as a prediction with 0 pred_margin
tf.app.flags.DEFINE_string('frozen', '1', "Frozen LSTM states")     # frozen encoded graph states for person Re-ID
tf.app.flags.DEFINE_string('pre_epochs', '200', "epochs for pre-training")
tf.app.flags.DEFINE_string('pre_train', '1', "pre-train or not")
tf.app.flags.DEFINE_string('s_range', 'all', "1, f/2, f-1, all")
tf.app.flags.DEFINE_string('pred_margin', '0', "0, 1, 2")
tf.app.flags.DEFINE_string('view', '', "test different views on CASIA B")
tf.app.flags.DEFINE_string('reverse', '0', "use reverse sequences")
tf.app.flags.DEFINE_string('bi', '0', "bi-directional prediction")
tf.app.flags.DEFINE_string('save_flag', '1', "save model or not")
tf.app.flags.DEFINE_string('consecutive_pre', '0', "consecutive_pre")
tf.app.flags.DEFINE_string('single_level', '0', "single_level")
tf.app.flags.DEFINE_string('global_att', '0', "global_att")
tf.app.flags.DEFINE_string('last_pre', '0', "last_pre")
tf.app.flags.DEFINE_string('random_sample', '0', "random_sample")
tf.app.flags.DEFINE_string('ord_sample', '0', "ord_sample")
tf.app.flags.DEFINE_string('concate', '0', "concate")
tf.app.flags.DEFINE_string('struct_only', '0', "struct_only")
tf.app.flags.DEFINE_string('abla', '0', "abla")
tf.app.flags.DEFINE_string('P', '8', "P")
tf.app.flags.DEFINE_string('n_hood', '1', "n_hood")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery")
tf.app.flags.DEFINE_string('no_MSR', '0', "using MSR or not")
tf.app.flags.DEFINE_string('patience', '100', "epochs for early stopping")
tf.app.flags.DEFINE_string('sample_num', '1', "sampling times")
tf.app.flags.DEFINE_string('fusion_lambda', '1', "collaboration fusion coefficient")
tf.app.flags.DEFINE_string('loss', 'l1', "use l1, l2 or MSE loss")

FLAGS = tf.app.flags.FLAGS

# check parameters
if FLAGS.dataset not in ['IAS', 'KGBD', 'KS20']:
	raise Exception('Dataset must be IAS, KGBD, or KS20.')
if not FLAGS.gpu.isdigit() or int(FLAGS.gpu) < 0:
	raise Exception('GPU number must be a positive integer.')
if FLAGS.length not in ['4', '6', '8', '10']:
	raise Exception('Length number must be 4, 6, 8 or 10.')
if FLAGS.split not in ['', 'A', 'B']:
	raise Exception('Datset split must be "A" (for IAS-A), "B" (for IAS-B), "" (for other datasets).')
if float(FLAGS.fusion_lambda) < 0 or float(FLAGS.fusion_lambda) > 1:
	raise Exception('Collaboration Fusion coefficient must be not less than 0 or not larger than 1.')
if FLAGS.pre_train not in ['1', '0']:
	raise Exception('Pre-train Flag must be 0 or 1.')
if FLAGS.save_flag not in ['1', '0']:
	raise Exception('Save_flag must be 0 or 1.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset
# optimal paramters
if dataset == 'KS20':
	batch_size = 64
	LSTM_embed = 256
	lr = 0.0005
elif dataset == 'IAS' and split == 'A':
	batch_size = 128
	LSTM_embed = 256
	lr = 0.0025
elif dataset == 'IAS' and split == 'B':
	batch_size = 64
	LSTM_embed = 256
	lr = 0.0025
elif dataset == 'CASIA_B':
	batch_size = 128
	LSTM_embed = 256
	lr = 0.0005
# 	patience = 50
if int(FLAGS.pre_epochs) != 200:
	pre_epochs = int(FLAGS.pre_epochs)
elif dataset == 'KGBD':
	batch_size = 256
	LSTM_embed = 128
	lr = 0.0025
	pre_epochs = 160
if dataset == 'IAS' or dataset == 'BIWI':
	pre_epochs = 120
if dataset == 'CASIA_B':
	pre_epochs = 160
	if FLAGS.probe_type != '':
		pre_epochs = 200
time_step = int(FLAGS.length)
k_values = list(range(1, time_step - int(FLAGS.pred_margin) + 1))
k_filter = -1
if FLAGS.s_range == '1':
	k_filter = 1
elif FLAGS.s_range == 'f/2':
	k_filter = int(time_step//2)
elif FLAGS.s_range == 'f-1':
	k_filter = time_step-1
elif FLAGS.s_range == 'f':
	k_filter = time_step

fusion_lambda = float(FLAGS.fusion_lambda)
split = FLAGS.split
pretext = FLAGS.task
levels = len(k_values)
frozen= FLAGS.frozen
nhood = int(FLAGS.n_hood)
pred_margin = FLAGS.pred_margin
s_range = FLAGS.s_range
save_flag = FLAGS.save_flag
patience = int(FLAGS.patience)
sample_num = int(FLAGS.sample_num)

consecutive_pre = False
single_level = False
global_att = False
MG_only = False
last_pre = False
ord_sample = False
concate = False
struct_only = False
abla = False
P = '8'

only_ske_embed = False
three_layer = False
embed_half = False
one_layer = False
CAGEs_embed = False
no_interp = False
no_multi_pred = False

change = '_formal'

if FLAGS.loss != 'l1':
	change += '_loss_' + FLAGS.loss

if sample_num != 1:
	change = '_sample_num_' + str(sample_num)
if FLAGS.probe_type != '':
	change = '_CME'
if FLAGS.fusion_lambda != '1':
	change = '_lambda_' + FLAGS.fusion_lambda

if FLAGS.consecutive_pre == '1':
	consecutive_pre = True
if FLAGS.single_level == '1':
	single_level = True
if FLAGS.global_att == '1':
	global_att = True
if FLAGS.last_pre == '1':
	last_pre = True
if FLAGS.ord_sample == '1':
	ord_sample = True
if FLAGS.concate == '1':
	concate = True
if FLAGS.struct_only == '1':
	struct_only = True
if FLAGS.P != '8':
	P = FLAGS.P
	Ps = [int(P), 1]

try:
	os.mkdir(RN_dir)
except:
	pass
try:
	os.mkdir(pre_dir)
except:
	pass

if consecutive_pre:
	pre_dir += '_consec_pre'
	RN_dir += '_consec_pre'
if single_level:
	fusion_lambda = 0
	pre_dir += '_single'
	RN_dir += '_single'
if global_att:
	pre_dir += '_global_att'
	RN_dir += '_global_att'
if MG_only:
	pre_dir += '_MG_only'
	RN_dir += '_MG_only'
if last_pre:
	pre_dir += '_last_pre'
	RN_dir += '_last_pre'
if ord_sample:
	pre_dir += '_ord_sample'
	RN_dir += '_ord_sample'
if concate:
	pre_dir += '_concate'
	RN_dir += '_concate'
if frozen == '1':
	pre_dir += '_frozen'
	RN_dir += '_frozen'
if struct_only:
	pre_dir += '_struct_only'
	RN_dir += '_struct_only'
if P != '8':
	pre_dir += '_P_' + P
	RN_dir += '_P_' + P

if pretext == 'none':
	pre_epochs = 0
I_nodes = 39
if dataset == 'KS20':
	nb_nodes = 25
	I_nodes = 49
if dataset == 'CASIA_B':
	nb_nodes = 14
	I_nodes = 27

if FLAGS.view != '':
	view_dir = '_view_' + FLAGS.view
else:
	view_dir = ''

print('Dataset: ' + dataset)

print('----- Opt. hyperparams -----')
print('pre_train_epochs: ' + str(pre_epochs))
print('nhood: ' + str(nhood))
print('skeleton_nodes: ' + str(nb_nodes))
print('seqence_length: ' + str(time_step))
print('pretext: ' + str(pretext))
print('fusion_lambda: ' + str(fusion_lambda))
print('batch_size: ' + str(batch_size))
print('lr: ' + str(lr))
print('view: ' + FLAGS.view)
print('P: ' + FLAGS.P)
print('fusion_lambda: ' + FLAGS.fusion_lambda)
print('loss_type: ' + FLAGS.loss)
print('patience: ' + FLAGS.patience)
print('save_flag: ' + FLAGS.save_flag)

print('----- Archi. hyperparams -----')
print('structural relation matrix number: ' + str(Ps[0]))
print('LSTM_embed_num: ' + str(LSTM_embed))
print('LSTM_layer_num: ' + str(num_layers))

"""
 Obtain training and testing data in hyper-joint-scale, joint-scale, part-scale, and body-scale.
 Generate corresponding adjacent matrix and bias.
"""
if FLAGS.probe_type == '':
	X_train_J, X_train_P, X_train_B, X_train_I, y_train, X_test_J, X_test_P, X_test_B, X_test_I, y_test, \
	adj_J, biases_J, adj_P, biases_P, adj_B, biases_B, adj_I, biases_I, nb_classes = \
		process.gen_train_data(dataset=dataset, split=split, time_step=time_step,
		                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, view=FLAGS.view, reverse=FLAGS.reverse)
else:
	from utils import process_cme as process
	X_train_J, X_train_P, X_train_B, X_train_I, y_train, X_test_J, X_test_P, X_test_B, X_test_I, y_test, \
	adj_J, biases_J, adj_P, biases_P, adj_B, biases_B, adj_I, biases_I, nb_classes = \
		process.gen_train_data(dataset=dataset, split=split, time_step=time_step,
		                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
		                      reverse=FLAGS.reverse, PG_type=FLAGS.probe_type.split('.')[0])
	print('## [Probe].[Gallery]', FLAGS.probe_type)


if FLAGS.pre_train == '1':
	with tf.Graph().as_default():
		with tf.name_scope('Input'):
			lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_classes))
			J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size*time_step, nb_nodes, ft_size))
			P_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 10, ft_size))
			B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 5, ft_size))
			# Interpolation
			I_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, I_nodes, ft_size))
			J_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
			P_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 10, 10))
			B_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5))
			I_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, I_nodes, I_nodes))
			attn_drop = tf.placeholder(dtype=tf.float32, shape=())
			ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
			is_train = tf.placeholder(dtype=tf.bool, shape=())

		with tf.name_scope("Multi_Scale"), tf.variable_scope("Multi_Scale", reuse=tf.AUTO_REUSE):
			def SRL(J_in, J_bias_in, nb_nodes):
				W_h = tf.Variable(tf.random_normal([3, hid_units[-1]]))
				b_h = tf.Variable(tf.zeros(shape=[hid_units[-1], ]))
				J_h = tf.reshape(J_in, [-1, ft_size])

				J_h = tf.matmul(J_h, W_h) + b_h
				J_h = tf.reshape(J_h, [batch_size*time_step, nb_nodes, hid_units[-1]])
				if not concate:
					J_seq_ftr = model.inference(J_h, 0, nb_nodes, is_train,
					                         attn_drop, ffd_drop,
					                         bias_mat=J_bias_in,
					                         hid_units=hid_units, n_heads=Ps,
					                         residual=residual, activation=nonlinearity, r_pool=True)
				else:
					J_seq_ftr = model.inference(J_h, 0, nb_nodes, is_train,
					                            attn_drop, ffd_drop,
					                            bias_mat=J_bias_in,
					                            hid_units=hid_units, n_heads=Ps,
					                            residual=residual, activation=nonlinearity, r_pool=False)
				return J_seq_ftr


			def CRL(s1, s2, s1_num, s2_num, hid_in):
				r_unorm = tf.matmul(s2, tf.transpose(s1, [0, 2, 1]))
				att_w = tf.nn.softmax(r_unorm)
				att_w = tf.expand_dims(att_w, axis=-1)
				s1 = tf.reshape(s1, [s1.shape[0], 1, s1.shape[1], hid_in])
				c_ftr = tf.reduce_sum(att_w * s1, axis=2)
				c_ftr = tf.reshape(c_ftr, [-1, hid_in])
				att_w = tf.reshape(att_w, [-1, s1_num * s2_num])
				return r_unorm, c_ftr


			def MGRN(J_in, P_in, B_in, I_in, J_bias_in, P_bias_in, B_bias_in, I_bias_in, hid_in, hid_out):
				h_J_seq_ftr = SRL(J_in=J_in, J_bias_in=J_bias_in, nb_nodes=nb_nodes)
				h_P_seq_ftr = SRL(J_in=P_in, J_bias_in=P_bias_in, nb_nodes=10)
				h_B_seq_ftr = SRL(J_in=B_in, J_bias_in=B_bias_in, nb_nodes=5)
				h_I_seq_ftr = SRL(J_in=I_in, J_bias_in=I_bias_in, nb_nodes=I_nodes)

				h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes, hid_in])
				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])
				h_I_seq_ftr = tf.reshape(h_I_seq_ftr, [-1, I_nodes, hid_in])

				W_cs_12 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_23 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_13 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_I = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_01 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_02 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_03 = tf.Variable(tf.random_normal([hid_in, hid_out]))


				W_self_0 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_self_1 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_self_2 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_self_3 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				self_a_0, self_r_0 = CRL(h_I_seq_ftr, h_I_seq_ftr, I_nodes, I_nodes, hid_in)
				self_a_1, self_r_1 = CRL(h_J_seq_ftr, h_J_seq_ftr, nb_nodes, nb_nodes, hid_in)
				self_a_2, self_r_2 = CRL(h_P_seq_ftr, h_P_seq_ftr, 10, 10, hid_in)
				self_a_3, self_r_3 = CRL(h_B_seq_ftr, h_B_seq_ftr, 5, 5, hid_in)

				h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, hid_in])
				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_in])
				h_I_seq_ftr = tf.reshape(h_I_seq_ftr, [-1, hid_in])


				h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes, hid_in])
				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])
				h_I_seq_ftr = tf.reshape(h_I_seq_ftr, [-1, I_nodes, hid_in])


				a_12, r_12 = CRL(h_P_seq_ftr, h_J_seq_ftr, 10, nb_nodes, hid_in)
				a_13, r_13 = CRL(h_B_seq_ftr, h_J_seq_ftr, 5, nb_nodes, hid_in)
				a_01, r_01 = CRL(h_J_seq_ftr, h_I_seq_ftr, nb_nodes, I_nodes, hid_in)
				a_02, r_02 = CRL(h_P_seq_ftr, h_I_seq_ftr, 10, I_nodes, hid_in)
				a_03, r_03 = CRL(h_B_seq_ftr, h_I_seq_ftr, 5, I_nodes, hid_in)
				a_23, r_23 = CRL(h_B_seq_ftr, h_P_seq_ftr, 5, 10, hid_in)



				h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, hid_in])
				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_in])
				h_I_seq_ftr = tf.reshape(h_I_seq_ftr, [-1, hid_in])

				if not struct_only:
					h_J_seq_ftr = h_J_seq_ftr + float(FLAGS.fusion_lambda) * (tf.matmul(self_r_1, W_self_1) + tf.matmul(r_12, W_cs_12) + tf.matmul(r_13, W_cs_13))
					h_I_seq_ftr = h_I_seq_ftr + float(FLAGS.fusion_lambda) * (tf.matmul(self_r_0, W_self_0) + tf.matmul(r_01, W_cs_01) + tf.matmul(r_02, W_cs_02) + tf.matmul(r_03, W_cs_03))
					h_P_seq_ftr = h_P_seq_ftr + float(FLAGS.fusion_lambda) * (tf.matmul(self_r_2, W_self_2) + tf.matmul(r_23, W_cs_23))
					h_B_seq_ftr = h_B_seq_ftr + float(FLAGS.fusion_lambda) * (tf.matmul(self_r_3, W_self_3))



				h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes,  hid_out])
				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10,  hid_out])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5,  hid_out])
				h_I_seq_ftr = tf.reshape(h_I_seq_ftr, [-1, I_nodes, hid_out])

				return h_B_seq_ftr, h_P_seq_ftr, h_J_seq_ftr, h_I_seq_ftr

			if not concate:
				h_B_seq_ftr, h_P_seq_ftr, h_J_seq_ftr, h_I_seq_ftr = MGRN(J_in, P_in, B_in,
				                                                                                I_in, J_bias_in,
				                                                                                P_bias_in,
				                                                                                B_bias_in,
				                                                                                I_bias_in,
				                                                                                hid_units[-1],
				                                                                                hid_units[-1])
			else:
				h_B_seq_ftr, h_P_seq_ftr, h_J_seq_ftr = MSGAM(J_in, P_in, B_in, J_bias_in, P_bias_in, B_bias_in,
				                                             hid_units[-1] * Ps[0], hid_units[-1] * Ps[0])
			h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, hid_units[-1]])
			h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_units[-1]])
			h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_units[-1]])
			h_I_seq_ftr = tf.reshape(h_I_seq_ftr, [-1, hid_units[-1]])

		ftr_in = J_in
		P_ftr_in = P_in
		B_ftr_in = B_in
		I_ftr_in = I_in

		J_seq_ftr = tf.reshape(h_J_seq_ftr, [batch_size, time_step, -1])
		P_seq_ftr = tf.reshape(h_P_seq_ftr, [batch_size, time_step, -1])
		B_seq_ftr = tf.reshape(h_B_seq_ftr, [batch_size, time_step, -1])
		I_seq_ftr = tf.reshape(h_I_seq_ftr, [batch_size, time_step, -1])

		J_T_m_ftr = []
		J_Pred_tar = []
		J_Test_ftr = []
		J_skes_in = tf.reshape(ftr_in, [batch_size, time_step, -1])
		# part-scale
		P_T_m_ftr = []
		P_Pred_tar = []
		P_Test_ftr = []
		P_skes_in = tf.reshape(P_ftr_in, [batch_size, time_step, -1])
		# body-scale
		B_T_m_ftr = []
		B_Pred_tar = []
		B_Test_ftr = []
		B_skes_in = tf.reshape(B_ftr_in, [batch_size, time_step, -1])
		# interpolation
		I_T_m_ftr = []
		I_Pred_tar = []
		I_Test_ftr = []
		I_skes_in = tf.reshape(I_ftr_in, [batch_size, time_step, -1])
		print_flag = False
		for i in range(sample_num):
			for k in range(1, len(k_values)+2):
				seq_ind = np.arange(batch_size).reshape(-1, 1)
				seq_ind = np.tile(seq_ind, [1, k]).reshape(-1, 1)
				if k <= time_step - int(pred_margin):
					# in order
					if ord_sample:
						T_m = np.arange(k).reshape(-1, 1)
					else:
						T_m = np.random.choice(time_step - int(pred_margin), size=[k], replace=False).reshape(-1, 1)
					T_m = np.sort(T_m, axis=0)
					if pretext == 'pre':
						Pred_t = T_m + int(pred_margin)
					T_m = T_m.astype(dtype=np.int32)
					Pred_t = Pred_t.astype(dtype=np.int32)
					T_m = np.tile(T_m, [batch_size, 1]).reshape(-1, 1)
					# T_m_tar.append(T_m)
					Pred_t = np.tile(Pred_t, [batch_size, 1]).reshape(-1, 1)

					T_m = np.hstack([seq_ind, T_m])
					Pred_t = np.hstack([seq_ind, Pred_t])
					J_sampled_seq_ftr = tf.gather_nd(J_seq_ftr, T_m)
					# P, B
					P_sampled_seq_ftr = tf.gather_nd(P_seq_ftr, T_m)
					B_sampled_seq_ftr = tf.gather_nd(B_seq_ftr, T_m)
					I_sampled_seq_ftr = tf.gather_nd(I_seq_ftr, T_m)
					J_Pred_t_seq = tf.gather_nd(J_skes_in, Pred_t)
					J_sampled_seq_ftr = tf.reshape(J_sampled_seq_ftr, [batch_size, k, -1])
					J_Pred_t_seq = tf.reshape(J_Pred_t_seq, [batch_size, k, -1])
					# P,B
					P_Pred_t_seq = tf.gather_nd(P_skes_in, Pred_t)
					P_sampled_seq_ftr = tf.reshape(P_sampled_seq_ftr, [batch_size, k, -1])
					P_Pred_t_seq = tf.reshape(P_Pred_t_seq, [batch_size, k, -1])
					B_Pred_t_seq = tf.gather_nd(B_skes_in, Pred_t)
					B_sampled_seq_ftr = tf.reshape(B_sampled_seq_ftr, [batch_size, k, -1])
					B_Pred_t_seq = tf.reshape(B_Pred_t_seq, [batch_size, k, -1])
					I_Pred_t_seq = tf.gather_nd(I_skes_in, Pred_t)
					I_sampled_seq_ftr = tf.reshape(I_sampled_seq_ftr, [batch_size, k, -1])
					I_Pred_t_seq = tf.reshape(I_Pred_t_seq, [batch_size, k, -1])

					J_T_m_ftr.append(J_sampled_seq_ftr)
					J_Pred_tar.append(J_Pred_t_seq)
					#
					P_T_m_ftr.append(P_sampled_seq_ftr)
					P_Pred_tar.append(P_Pred_t_seq)
					#
					B_T_m_ftr.append(B_sampled_seq_ftr)
					B_Pred_tar.append(B_Pred_t_seq)
					I_T_m_ftr.append(I_sampled_seq_ftr)
					I_Pred_tar.append(I_Pred_t_seq)

				if i == 0:
					T_m_test = np.arange(k).reshape(-1, 1)
					# print(T_m_test)
					T_m_test = T_m_test.astype(dtype=np.int32)
					T_m_test = np.tile(T_m_test, [batch_size, 1]).reshape(-1, 1)
					T_m_test = np.hstack([seq_ind, T_m_test])
					J_test_seq_ftr = tf.gather_nd(J_seq_ftr, T_m_test)
					J_test_seq_ftr = tf.reshape(J_test_seq_ftr, [batch_size, k, -1])
					J_Test_ftr.append(J_test_seq_ftr)
					#
					P_test_seq_ftr = tf.gather_nd(P_seq_ftr, T_m_test)
					P_test_seq_ftr = tf.reshape(P_test_seq_ftr, [batch_size, k, -1])
					P_Test_ftr.append(P_test_seq_ftr)
					B_test_seq_ftr = tf.gather_nd(B_seq_ftr, T_m_test)
					B_test_seq_ftr = tf.reshape(B_test_seq_ftr, [batch_size, k, -1])
					B_Test_ftr.append(B_test_seq_ftr)
					I_test_seq_ftr = tf.gather_nd(I_seq_ftr, T_m_test)
					I_test_seq_ftr = tf.reshape(I_test_seq_ftr, [batch_size, k, -1])
					I_Test_ftr.append(I_test_seq_ftr)



		if FLAGS.bi == '1':
			for i in range(sample_num):
				for k in range(1, len(k_values) + 2):
					seq_ind = np.arange(batch_size).reshape(-1, 1)
					seq_ind = np.tile(seq_ind, [1, k]).reshape(-1, 1)
					if k <= time_step - int(pred_margin):
						# in order
						if ord_sample:
							T_m = np.arange(k).reshape(-1, 1)
						else:
							T_m = np.random.choice(time_step - int(pred_margin), size=[k], replace=False).reshape(-1, 1)
						# print (T_m)
						# print(T_m.shape)
						# if not random_sample:
						T_m = np.sort(T_m, axis=0)
						# Reverse
						T_m = np.sort(-T_m)
						T_m += time_step
						if pretext == 'pre':
							Pred_t = T_m - int(pred_margin)
						# 	no used
						# elif pretext == 'recon':
						# 	Pred_t = T_m
						# elif pretext == 'rev':
						# 	Pred_t = np.sort(-T_m)
						# 	Pred_t = -Pred_t
						print(T_m)
						print(Pred_t)
						T_m = T_m.astype(dtype=np.int32)
						Pred_t = Pred_t.astype(dtype=np.int32)
						T_m = np.tile(T_m, [batch_size]).reshape(-1, 1)
						# T_m_tar.append(T_m)
						Pred_t = np.tile(Pred_t, [batch_size]).reshape(-1, 1)

						T_m = np.hstack([seq_ind, T_m])
						Pred_t = np.hstack([seq_ind, Pred_t])
						J_sampled_seq_ftr = tf.gather_nd(J_seq_ftr, T_m)
						# P, B
						P_sampled_seq_ftr = tf.gather_nd(P_seq_ftr, T_m)
						B_sampled_seq_ftr = tf.gather_nd(B_seq_ftr, T_m)
						I_sampled_seq_ftr = tf.gather_nd(I_seq_ftr, T_m)
						#
						J_Pred_t_seq = tf.gather_nd(J_skes_in, Pred_t)
						J_sampled_seq_ftr = tf.reshape(J_sampled_seq_ftr, [batch_size, k, -1])
						J_Pred_t_seq = tf.reshape(J_Pred_t_seq, [batch_size, k, -1])
						# P,B
						P_Pred_t_seq = tf.gather_nd(P_skes_in, Pred_t)
						P_sampled_seq_ftr = tf.reshape(P_sampled_seq_ftr, [batch_size, k, -1])
						P_Pred_t_seq = tf.reshape(P_Pred_t_seq, [batch_size, k, -1])
						B_Pred_t_seq = tf.gather_nd(B_skes_in, Pred_t)
						B_sampled_seq_ftr = tf.reshape(B_sampled_seq_ftr, [batch_size, k, -1])
						B_Pred_t_seq = tf.reshape(B_Pred_t_seq, [batch_size, k, -1])
						I_Pred_t_seq = tf.gather_nd(I_skes_in, Pred_t)
						I_sampled_seq_ftr = tf.reshape(I_sampled_seq_ftr, [batch_size, k, -1])
						I_Pred_t_seq = tf.reshape(I_Pred_t_seq, [batch_size, k, -1])
						#
						# sorted random frames
						J_T_m_ftr.append(J_sampled_seq_ftr)
						J_Pred_tar.append(J_Pred_t_seq)
						#
						P_T_m_ftr.append(P_sampled_seq_ftr)
						P_Pred_tar.append(P_Pred_t_seq)
						#
						B_T_m_ftr.append(B_sampled_seq_ftr)
						B_Pred_tar.append(B_Pred_t_seq)
						I_T_m_ftr.append(I_sampled_seq_ftr)
						I_Pred_tar.append(I_Pred_t_seq)

		if FLAGS.bi == '1':
			sample_num = sample_num * 2

		with tf.name_scope("MSR"), tf.variable_scope("MSR", reuse=tf.AUTO_REUSE):
			J_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(LSTM_embed) for _ in range(num_layers)])
			P_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(LSTM_embed) for _ in range(num_layers)])
			B_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(LSTM_embed) for _ in range(num_layers)])
			I_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(LSTM_embed) for _ in range(num_layers)])

			J_all_pred_loss = []
			# P, B
			P_all_pred_loss = []
			B_all_pred_loss = []
			I_all_pred_loss = []
			#
			a0_all_pred_loss = []
			a1_all_pred_loss = []
			a2_all_pred_loss = []
			# all_mask_loss = []
			J_en_outs = []
			J_en_outs_whole = []
			J_en_outs_test = []
			#
			P_en_outs = []
			P_en_outs_whole = []
			P_en_outs_test = []
			B_en_outs = []
			B_en_outs_whole = []
			B_en_outs_test = []
			I_en_outs = []
			I_en_outs_whole = []
			I_en_outs_test = []
			#
			ske_en_outs = []
			# all_mask_acc = []
			with tf.name_scope("J_pred"), tf.variable_scope("J_pred", reuse=tf.AUTO_REUSE):
				J_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				J_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				J_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, nb_nodes * 3]))
				J_b2_pred = tf.Variable(tf.zeros(shape=[nb_nodes * 3, ]))
				P_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				P_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				P_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 10 * 3]))
				P_b2_pred = tf.Variable(tf.zeros(shape=[10 * 3, ]))
				B_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				B_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				B_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 5 * 3]))
				B_b2_pred = tf.Variable(tf.zeros(shape=[5 * 3, ]))


				for i in range((levels - int(pred_margin)) * sample_num):
					J_sampled_seq_ftr = J_T_m_ftr[i]
					J_encoder_output, J_encoder_state = tf.nn.dynamic_rnn(J_cell, J_sampled_seq_ftr, dtype=tf.float32)

					J_encoder_output = tf.reshape(J_encoder_output, [-1, LSTM_embed])
					# Skeleton-level Prediction
					J_pred_embedding_J = tf.nn.relu(tf.matmul(J_encoder_output, J_W1_pred) + J_b1_pred)
					#
					J_pred_skeleton = tf.matmul(J_pred_embedding_J, J_W2_pred) + J_b2_pred

					J_pred_skeleton = tf.reshape(J_pred_skeleton, [batch_size, k_values[i % levels], nb_nodes * 3])


					# P, B
					P_pred_embedding_J = tf.nn.relu(tf.matmul(J_encoder_output, P_W1_pred) + P_b1_pred)
					P_pred_skeleton = tf.matmul(P_pred_embedding_J, P_W2_pred) + P_b2_pred
					P_pred_skeleton = tf.reshape(P_pred_skeleton, [batch_size, k_values[i % levels], 10 * 3])
					B_pred_embedding_J = tf.nn.relu(tf.matmul(J_encoder_output, B_W1_pred) + B_b1_pred)
					B_pred_skeleton = tf.matmul(B_pred_embedding_J, B_W2_pred) + B_b2_pred
					B_pred_skeleton = tf.reshape(B_pred_skeleton, [batch_size, k_values[i % levels], 5 * 3])
					if FLAGS.loss == 'l1':
						J_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(J_pred_skeleton, J_Pred_tar[i]))
						# P, B
						# tf.nn.l1_loss
						P_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(P_pred_skeleton, P_Pred_tar[i]))
						B_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(B_pred_skeleton, B_Pred_tar[i]))
					elif FLAGS.loss == 'MSE':
						J_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(J_pred_skeleton, J_Pred_tar[i]))
						# P, B
						# tf.nn.l1_loss
						P_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(P_pred_skeleton, P_Pred_tar[i]))
						B_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(B_pred_skeleton, B_Pred_tar[i]))
					elif FLAGS.loss == 'l2':
						J_pred_loss = tf.reduce_mean(tf.nn.l2_loss(J_pred_skeleton - J_Pred_tar[i]))
						# P, B
						# tf.nn.l1_loss
						P_pred_loss = tf.reduce_mean(tf.nn.l2_loss(P_pred_skeleton - P_Pred_tar[i]))
						B_pred_loss = tf.reduce_mean(tf.nn.l2_loss(B_pred_skeleton - B_Pred_tar[i]))

					if k_filter == -1 or (k_filter !=-1 and i == 0):
						if no_multi_pred:
							J_all_pred_loss.append(J_pred_loss)
						else:
							J_all_pred_loss.append(J_pred_loss+P_pred_loss+B_pred_loss)


					J_encoder_output = tf.reshape(J_encoder_output, [batch_size, k_values[i % levels], -1])
					# Average
					J_en_outs.append(J_encoder_output[:, -1, :])
					# en_outs.append(encoder_output)
					# if i == sample_num * levels - 1:
					# 	J_en_outs_whole = J_encoder_output
				for i in range(levels+1):
					J_test_seq_ftr = J_Test_ftr[i]
					J_encoder_output, J_encoder_state = tf.nn.dynamic_rnn(J_cell, J_test_seq_ftr, dtype=tf.float32)
					J_encoder_output = tf.reshape(J_encoder_output, [-1, LSTM_embed])
					# Skeleton-level Prediction
					J_encoder_output = tf.reshape(J_encoder_output, [batch_size, i+1, -1])
					# en_outs_test.append(encoder_output)
					J_en_outs_test.append(J_encoder_output[:, -1, :])

			with tf.name_scope("P_pred"), tf.variable_scope("P_pred", reuse=tf.AUTO_REUSE):
				P_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				P_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				P_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 10 * 3]))
				P_b2_pred = tf.Variable(tf.zeros(shape=[10 * 3, ]))
				B_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				B_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				B_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 5 * 3]))
				B_b2_pred = tf.Variable(tf.zeros(shape=[5 * 3, ]))
				for i in range((levels - int(pred_margin)) * sample_num):
					# if k_filter !=-1 and i != 0:
					# 	continue
					P_sampled_seq_ftr = P_T_m_ftr[i]
					P_encoder_output, P_encoder_state = tf.nn.dynamic_rnn(P_cell, P_sampled_seq_ftr, dtype=tf.float32)

					P_encoder_output = tf.reshape(P_encoder_output, [-1, LSTM_embed])
					# Skeleton-level Prediction
					P_pred_embedding_P = tf.nn.relu(tf.matmul(P_encoder_output, P_W1_pred) + P_b1_pred)

					P_pred_skeleton = tf.matmul(P_pred_embedding_P, P_W2_pred) + P_b2_pred

					P_pred_skeleton = tf.reshape(P_pred_skeleton, [batch_size, k_values[i % levels], 10 * 3])

					B_pred_embedding_P = tf.nn.relu(tf.matmul(P_encoder_output, B_W1_pred) + B_b1_pred)
					B_pred_skeleton = tf.matmul(B_pred_embedding_P, B_W2_pred) + B_b2_pred
					B_pred_skeleton = tf.reshape(B_pred_skeleton, [batch_size, k_values[i % levels], 5 * 3])

					if k_filter == -1 or (k_filter !=-1 and i == 0):
						if FLAGS.loss == 'l1':
							P_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(P_pred_skeleton, P_Pred_tar[i]))
							B_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(B_pred_skeleton, B_Pred_tar[i]))
						elif FLAGS.loss == 'MSE':
							P_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(P_pred_skeleton, P_Pred_tar[i]))
							B_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(B_pred_skeleton, B_Pred_tar[i]))
						elif FLAGS.loss == 'l2':
							P_pred_loss = tf.reduce_mean(tf.nn.l2_loss(P_pred_skeleton - P_Pred_tar[i]))
							B_pred_loss = tf.reduce_mean(tf.nn.l2_loss(B_pred_skeleton - B_Pred_tar[i]))
					if no_multi_pred:
						P_all_pred_loss.append(P_pred_loss)
					else:
						P_all_pred_loss.append(P_pred_loss+B_pred_loss)
					P_encoder_output = tf.reshape(P_encoder_output, [batch_size, k_values[i % levels], -1])
					# Average
					P_en_outs.append(P_encoder_output[:, -1, :])

				for i in range(levels + 1):
					P_test_seq_ftr = P_Test_ftr[i]
					P_encoder_output, P_encoder_state = tf.nn.dynamic_rnn(P_cell, P_test_seq_ftr, dtype=tf.float32)
					P_encoder_output = tf.reshape(P_encoder_output, [-1, LSTM_embed])
					P_encoder_output = tf.reshape(P_encoder_output, [batch_size, i+1, -1])
					P_en_outs_test.append(P_encoder_output[:, -1, :])
			with tf.name_scope("B_pred"), tf.variable_scope("B_pred", reuse=tf.AUTO_REUSE):
				B_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				B_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				B_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 5 * 3]))
				B_b2_pred = tf.Variable(tf.zeros(shape=[5 * 3, ]))

				for i in range((levels - int(pred_margin)) * sample_num):
					B_sampled_seq_ftr = B_T_m_ftr[i]
					B_encoder_output, B_encoder_state = tf.nn.dynamic_rnn(B_cell, B_sampled_seq_ftr, dtype=tf.float32)

					B_encoder_output = tf.reshape(B_encoder_output, [-1, LSTM_embed])
					# Skeleton-level Prediction
					B_pred_embedding_B = tf.nn.relu(tf.matmul(B_encoder_output, B_W1_pred) + B_b1_pred)


					B_pred_skeleton = tf.matmul(B_pred_embedding_B, B_W2_pred) + B_b2_pred
					B_pred_skeleton = tf.reshape(B_pred_skeleton, [batch_size, k_values[i % levels], 5 * 3])
					# tf.nn.l1_loss
					if k_filter == -1 or (k_filter !=-1 and i == 0):
						if FLAGS.loss == 'l1':
							B_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(B_pred_skeleton, B_Pred_tar[i]))
						elif FLAGS.loss == 'MSE':
							B_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(B_pred_skeleton, B_Pred_tar[i]))
						elif FLAGS.loss == 'l2':
							B_pred_loss = tf.reduce_mean(tf.nn.l2_loss(B_pred_skeleton - B_Pred_tar[i]))
					B_all_pred_loss.append(B_pred_loss)
					B_encoder_output = tf.reshape(B_encoder_output, [batch_size, k_values[i % levels], -1])
					# Average
					B_en_outs.append(B_encoder_output[:, -1, :])

				for i in range(levels + 1):
					B_test_seq_ftr = B_Test_ftr[i]
					B_encoder_output, B_encoder_state = tf.nn.dynamic_rnn(B_cell, B_test_seq_ftr, dtype=tf.float32)
					B_encoder_output = tf.reshape(B_encoder_output, [-1, LSTM_embed])
					B_encoder_output = tf.reshape(B_encoder_output, [batch_size, i+1, -1])
					B_en_outs_test.append(B_encoder_output[:, -1, :])
			with tf.name_scope("I_pred"), tf.variable_scope("I_pred", reuse=tf.AUTO_REUSE):
				J_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				J_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				J_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, nb_nodes * 3]))
				J_b2_pred = tf.Variable(tf.zeros(shape=[nb_nodes * 3, ]))
				P_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				P_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				P_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 10 * 3]))
				P_b2_pred = tf.Variable(tf.zeros(shape=[10 * 3, ]))
				B_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				B_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				B_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, 5 * 3]))
				B_b2_pred = tf.Variable(tf.zeros(shape=[5 * 3, ]))
				I_W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				I_b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				I_W2_pred = tf.Variable(tf.random_normal([LSTM_embed, I_nodes * 3]))
				I_b2_pred = tf.Variable(tf.zeros(shape=[I_nodes * 3, ]))
				for i in range((levels - int(pred_margin)) * sample_num):
					# if k_filter == time_step and i != 0:
					# 	continue
					I_sampled_seq_ftr = I_T_m_ftr[i]

					I_encoder_output, I_encoder_state = tf.nn.dynamic_rnn(I_cell, I_sampled_seq_ftr, dtype=tf.float32)
					I_encoder_output = tf.reshape(I_encoder_output, [-1, LSTM_embed])
					# Skeleton-level Prediction
					I_pred_embedding_I = tf.nn.relu(tf.matmul(I_encoder_output, I_W1_pred) + I_b1_pred)

					I_pred_skeleton = tf.matmul(I_pred_embedding_I, I_W2_pred) + I_b2_pred
					I_pred_skeleton = tf.reshape(I_pred_skeleton, [batch_size, k_values[i % levels], I_nodes * 3])
					# tf.nn.l1_loss
					J_pred_embedding_I = tf.nn.relu(tf.matmul(I_encoder_output, J_W1_pred) + J_b1_pred)
					J_pred_skeleton = tf.matmul(J_pred_embedding_I, J_W2_pred) + J_b2_pred
					J_pred_skeleton = tf.reshape(J_pred_skeleton, [batch_size, k_values[i % levels], nb_nodes * 3])

					B_pred_embedding_I = tf.nn.relu(tf.matmul(I_encoder_output, B_W1_pred) + B_b1_pred)
					B_pred_skeleton = tf.matmul(B_pred_embedding_I, B_W2_pred) + B_b2_pred
					B_pred_skeleton = tf.reshape(B_pred_skeleton, [batch_size, k_values[i % levels], 5 * 3])
					P_pred_embedding_I = tf.nn.relu(tf.matmul(I_encoder_output, P_W1_pred) + P_b1_pred)
					P_pred_skeleton = tf.matmul(P_pred_embedding_I, P_W2_pred) + P_b2_pred
					P_pred_skeleton = tf.reshape(P_pred_skeleton, [batch_size, k_values[i % levels], 10 * 3])
					#
					if k_filter == -1 or (k_filter !=-1 and i == 0):
						if FLAGS.loss == 'l1':
							I_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(I_pred_skeleton, I_Pred_tar[i]))
							J_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(J_pred_skeleton, J_Pred_tar[i]))
							P_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(P_pred_skeleton, P_Pred_tar[i]))
							B_pred_loss = tf.reduce_mean(tf.losses.absolute_difference(B_pred_skeleton, B_Pred_tar[i]))
						elif FLAGS.loss == 'MSE':
							I_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(I_pred_skeleton, I_Pred_tar[i]))
							J_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(J_pred_skeleton, J_Pred_tar[i]))
							P_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(P_pred_skeleton, P_Pred_tar[i]))
							B_pred_loss = tf.reduce_mean(tf.losses.mean_squared_error(B_pred_skeleton, B_Pred_tar[i]))
						elif FLAGS.loss == 'l2':
							I_pred_loss = tf.reduce_mean(tf.nn.l2_loss(I_pred_skeleton - I_Pred_tar[i]))
							J_pred_loss = tf.reduce_mean(tf.nn.l2_loss(J_pred_skeleton - J_Pred_tar[i]))
							P_pred_loss = tf.reduce_mean(tf.nn.l2_loss(P_pred_skeleton - P_Pred_tar[i]))
							B_pred_loss = tf.reduce_mean(tf.nn.l2_loss(B_pred_skeleton - B_Pred_tar[i]))
					if no_multi_pred:
						I_all_pred_loss.append(I_pred_loss)
					else:
						I_all_pred_loss.append(P_pred_loss + J_pred_loss + I_pred_loss + B_pred_loss)
					I_encoder_output = tf.reshape(I_encoder_output, [batch_size, k_values[i % levels], -1])
					# Average
					I_en_outs.append(I_encoder_output[:, -1, :])
				for i in range(levels + 1):
					I_test_seq_ftr = I_Test_ftr[i]
					I_encoder_output, I_encoder_state = tf.nn.dynamic_rnn(I_cell, I_test_seq_ftr, dtype=tf.float32)
					I_encoder_output = tf.reshape(I_encoder_output, [-1, LSTM_embed])
					I_encoder_output = tf.reshape(I_encoder_output, [batch_size, i+1, -1])
					I_en_outs_test.append(I_encoder_output[:, -1, :])
				# en_outs.append(tf.reshape(encoder_output, [batch_size, k_values[i], -1]))
			J_pred_opt = tf.train.AdamOptimizer(learning_rate=lr)
			J_pred_train_op = J_pred_opt.minimize(tf.reduce_mean(J_all_pred_loss))
			P_pred_opt = tf.train.AdamOptimizer(learning_rate=lr)
			P_pred_train_op = P_pred_opt.minimize(tf.reduce_mean(P_all_pred_loss))
			B_pred_opt = tf.train.AdamOptimizer(learning_rate=lr)
			B_pred_train_op = B_pred_opt.minimize(tf.reduce_mean(B_all_pred_loss))
			I_pred_opt = tf.train.AdamOptimizer(learning_rate=lr)
			I_pred_train_op = I_pred_opt.minimize(tf.reduce_mean(I_all_pred_loss))


		with tf.name_scope("Recognition"), tf.variable_scope("Recognition", reuse=tf.AUTO_REUSE):
			#
			if no_interp:
				en_to_pred = tf.concat([J_en_outs_test[0], P_en_outs_test[0],
				                        B_en_outs_test[0]], axis=-1)
				for i in range(1, levels + 1):
					temp = tf.concat([J_en_outs_test[i], P_en_outs_test[i],
					                  B_en_outs_test[i]], axis=-1)
					en_to_pred = tf.concat([en_to_pred, temp], axis=0)
			elif not only_ske_embed:
				en_to_pred = tf.concat([I_en_outs_test[0], J_en_outs_test[0], P_en_outs_test[0],
				                        B_en_outs_test[0]], axis=-1)
				for i in range(1, levels+1):
					temp = tf.concat([I_en_outs_test[i], J_en_outs_test[i], P_en_outs_test[i],
					                  B_en_outs_test[i]], axis=-1)
					en_to_pred = tf.concat([en_to_pred, temp], axis=0)
			else:
				en_to_pred = tf.concat([J_en_outs_test[0]], axis=-1)
				for i in range(1, levels + 1):
					temp = tf.concat([J_en_outs_test[i]], axis=-1)
					en_to_pred = tf.concat([en_to_pred, temp], axis=0)

			# Frozen
			if frozen == '1':
				en_to_pred = tf.stop_gradient(en_to_pred)

			if no_interp:
				# original
				W_1 = tf.Variable(tf.random_normal([LSTM_embed * 3, LSTM_embed * 3]))
				b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed * 3, ]))
				W_2 = tf.Variable(tf.random_normal([LSTM_embed * 3, nb_classes]))
				b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))
			elif not only_ske_embed:
				# original
				W_1 = tf.Variable(tf.random_normal([LSTM_embed * 4, LSTM_embed * 4]))
				b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed * 4, ]))
				W_2 = tf.Variable(tf.random_normal([LSTM_embed * 4, nb_classes]))
				b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))
			else:
				W_1 = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
				b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
				W_2 = tf.Variable(tf.random_normal([LSTM_embed, nb_classes]))
				b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))

			# original
			logits = tf.matmul(tf.nn.relu(tf.matmul(en_to_pred, W_1) + b_1), W_2) + b_2
			logits_pred = tf.matmul(tf.nn.relu(tf.matmul(en_to_pred, W_1) + b_1), W_2) + b_2


			log_resh = tf.reshape(logits, [-1, nb_classes])
			lab_resh = tf.reshape(lbl_in, [-1, nb_classes])


			if not last_pre:
				aver_pred = logits[:batch_size]
				aver_final_pred = logits_pred[:batch_size]
				for i in range(1, levels+1):
					aver_pred += logits[batch_size*i:batch_size*(i+1)]
					aver_final_pred += logits_pred[batch_size * i:batch_size * (i + 1)]
			else:
				aver_pred = logits[-batch_size:]
				aver_final_pred = logits_pred[-batch_size:]

			correct_pred = tf.equal(tf.argmax(aver_pred, -1), tf.argmax(lab_resh, -1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

			correct_final_pred = tf.equal(tf.argmax(aver_final_pred, -1), tf.argmax(lab_resh, -1))
			accuracy_final = tf.reduce_mean(tf.cast(correct_final_pred, tf.float32))

			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=aver_pred, labels=lab_resh))

			# frozen
			opt = tf.train.AdamOptimizer(learning_rate=lr)
			train_op = opt.minimize(loss)

		saver = tf.train.Saver()

		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		vlss_mn = np.inf
		vacc_mx = 0.0
		vnAUC_mx = 0.0
		curr_step = 0

		X_train = X_train_J
		X_test = X_test_J
		with tf.Session(config=config) as sess:
			sess.run(init_op)

			train_loss_avg = 0
			train_acc_avg = 0
			val_loss_avg = 0
			val_acc_avg = 0

			for epoch in range(pre_epochs):
				tr_step = 0
				tr_size = X_train.shape[0]
				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
					X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_input_B.reshape([-1, 5, 3])
					# interpolation
					X_input_I = X_train_I[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_I = X_input_I.reshape([-1, I_nodes, 3])
					loss_rec, loss_attr, acc_attr, loss_pred = 0, 0, 0, 0
					if no_interp:
						_, _, _, loss_pred, P_loss_pred, B_loss_pred, I_loss_pred, \
							= sess.run([J_pred_train_op, P_pred_train_op,
							            B_pred_train_op,
							            J_pred_loss, P_pred_loss, B_pred_loss, I_pred_loss,
							            ],
							           feed_dict={
								           J_in: X_input_J,
								           P_in: X_input_P,
								           B_in: X_input_B,
								           I_in: X_input_I,
								           J_bias_in: biases_J,
								           P_bias_in: biases_P,
								           B_bias_in: biases_B,
								           I_bias_in: biases_I,
								           lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
								           is_train: True,
								           attn_drop: 0.0, ffd_drop: 0.0})
					else:
						_, _, _, _, loss_pred, P_loss_pred, B_loss_pred, I_loss_pred, \
								 = sess.run([J_pred_train_op, P_pred_train_op,
						                                    B_pred_train_op, I_pred_train_op,
						                                                          J_pred_loss, P_pred_loss, B_pred_loss, I_pred_loss,
						                                                           ],
						                                                 feed_dict={
							                                                 J_in: X_input_J,
							                                                 P_in: X_input_P,
							                                                 B_in: X_input_B,
							                                                 I_in: X_input_I,
							                                                 J_bias_in: biases_J,
							                                                 P_bias_in: biases_P,
							                                                 B_bias_in: biases_B,
							                                                 I_bias_in: biases_I,
							                                                 lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
							                                                 is_train: True,
							                                                 attn_drop: 0.0, ffd_drop: 0.0})
					tr_step += 1
				print('[%s / %s] Train Loss (I. Pre.): %.5f | (J. Pre.): %.5f | (P. Pre.): %.5f | (B. Pre.): %.5f' %
				      (str(epoch), str(pre_epochs), I_loss_pred, loss_pred, P_loss_pred, B_loss_pred))
				if (epoch + 1) % 40 == 0:
					checkpt_file = pre_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(
						time_step) + \
					               '-' + dataset + '/' + split + \
					               '_' + str(epoch + 1) + '_' + str(s_range) + '_' + str(pred_margin) + '_' + FLAGS.reverse \
					               + view_dir + change + '.ckpt'
					print(checkpt_file)
					saver.save(sess, checkpt_file)
			checkpt_file = pre_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(
				time_step) + \
			               '-' + dataset + '/' + split + \
			               '_' + str(pre_epochs) + '_' + str(s_range) + '_' + str(pred_margin) + '_' + FLAGS.reverse +\
			               view_dir + change + '.ckpt'
			saver.save(sess, checkpt_file)

if split == 'B' and dataset == 'IAS':
		checkpt_file = pre_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(time_step) + \
	               '-' + dataset + '/A'  + \
	               '_' + str(pre_epochs) + '_' + str(s_range) + '_' + str(pred_margin) + '_' + FLAGS.reverse + view_dir +  change +  '.ckpt'
else:
	checkpt_file = pre_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(time_step) + \
	               '-' + dataset + '/' + split + \
	               '_' + str(pre_epochs) + '_' + str(s_range) + '_' + str(pred_margin) + '_' + FLAGS.reverse + view_dir +  change + '.ckpt'
print(checkpt_file)
# if FLAGS.pre_train == '1':
# 	saver.save(sess, checkpt_file)

# lr = 0.0005
if dataset == 'CASIA_B':
	lr = 0.0005
else:
	lr = 0.0025
loaded_graph = tf.Graph()
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session(graph=loaded_graph, config=config) as sess:
	loader = tf.train.import_meta_graph(checkpt_file + '.meta')
	# loader.restore(sess, checkpt_file)
	J_in = loaded_graph.get_tensor_by_name("Input/Placeholder_1:0")
	P_in = loaded_graph.get_tensor_by_name("Input/Placeholder_2:0")
	B_in = loaded_graph.get_tensor_by_name("Input/Placeholder_3:0")
	I_in = loaded_graph.get_tensor_by_name("Input/Placeholder_4:0")
	J_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_5:0")
	P_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_6:0")
	B_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_7:0")
	I_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_8:0")
	lbl_in = loaded_graph.get_tensor_by_name("Input/Placeholder:0")
	is_train = loaded_graph.get_tensor_by_name("Input/Placeholder_11:0")
	attn_drop = loaded_graph.get_tensor_by_name("Input/Placeholder_9:0")
	ffd_drop = loaded_graph.get_tensor_by_name("Input/Placeholder_10:0")
	en_to_pred = loaded_graph.get_tensor_by_name("Recognition/Recognition/StopGradient:0")

	# print(I_en_outs_test, J_en_outs_test, P_en_outs_test, B_en_outs_test)
	# exit()
	with tf.name_scope("Recognition"), tf.variable_scope("Recognition", reuse=tf.AUTO_REUSE):
		if frozen == '1':
			# en_to_loss = tf.stop_gradient(en_to_loss)
			en_to_pred = tf.stop_gradient(en_to_pred)
		if no_interp:
			W_1 = tf.Variable(tf.random_normal([LSTM_embed * 3, LSTM_embed * 3]))
			b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed * 3, ]))
		elif not only_ske_embed:
			# original
			W_1 = tf.Variable(tf.random_normal([LSTM_embed * 4, LSTM_embed * 4]))
			b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed * 4, ]))
		else:
			W_1 = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
			b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))

		if no_interp:
			W_2 = tf.Variable(tf.random_normal([LSTM_embed * 3, nb_classes]))
			b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))
		elif not only_ske_embed:
			W_2 = tf.Variable(tf.random_normal([LSTM_embed * 4, nb_classes]))
			b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))
		else:
			W_2 = tf.Variable(tf.random_normal([LSTM_embed, nb_classes]))
			b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))

		logits = tf.matmul(tf.nn.relu(tf.matmul(en_to_pred, W_1) + b_1), W_2) + b_2
		logits_pred = tf.matmul(tf.nn.relu(tf.matmul(en_to_pred, W_1) + b_1), W_2) + b_2

		log_resh = tf.reshape(logits, [-1, nb_classes])
		lab_resh = tf.reshape(lbl_in, [-1, nb_classes])

		if not last_pre:
			aver_pred = logits[:batch_size]
			aver_final_pred = logits_pred[:batch_size]

			# Multi-Scale
			for i in range(1, levels + 1):
				aver_pred += logits[batch_size * i:batch_size * (i + 1)]
				aver_final_pred += logits_pred[batch_size * i:batch_size * (i + 1)]
		else:
			aver_pred = logits[-batch_size:]
			aver_final_pred = logits_pred[-batch_size:]

		correct_pred = tf.equal(tf.argmax(aver_pred, -1), tf.argmax(lab_resh, -1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		correct_final_pred = tf.equal(tf.argmax(aver_final_pred, -1), tf.argmax(lab_resh, -1))
		accuracy_final = tf.reduce_mean(tf.cast(correct_final_pred, tf.float32))

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=aver_pred, labels=lab_resh))
		# print(aver_pred, accuracy_final, loss)
		# exit()
		opt = tf.train.AdamOptimizer(learning_rate=lr)
		train_op = opt.minimize(loss)
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	loader.restore(sess, checkpt_file)
	saver = tf.train.Saver()
	vlss_mn = np.inf
	vacc_mx = 0.0
	vnAUC_mx = 0.0
	curr_step = 0
	X_train = X_train_J
	X_test = X_test_J
	train_loss_avg = 0
	train_acc_avg = 0
	val_loss_avg = 0
	val_acc_avg = 0
	for epoch in range(nb_epochs):
		tr_step = 0
		tr_size = X_train.shape[0]
		train_acc_avg_final = 0
		logits_all = []
		labels_all = []
		while tr_step * batch_size < tr_size:
			if (tr_step + 1) * batch_size > tr_size:
				break
			X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
			X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
			X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
			X_input_P = X_input_P.reshape([-1, 10, 3])
			X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
			X_input_B = X_input_B.reshape([-1, 5, 3])
			X_input_I = X_train_I[tr_step * batch_size:(tr_step + 1) * batch_size]
			X_input_I = X_input_I.reshape([-1, I_nodes, 3])
			_, loss_value_tr, acc_tr, acc_tr_final, logits, labels = sess.run([train_op, loss, accuracy, accuracy_final, aver_final_pred, lab_resh],
			                  feed_dict={
				                  J_in: X_input_J,
				                  P_in: X_input_P,
				                  B_in: X_input_B,
				                  I_in: X_input_I,
				                  J_bias_in: biases_J,
				                  P_bias_in: biases_P,
				                  B_bias_in: biases_B,
				                  I_bias_in: biases_I,
				                  lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
				                  is_train: True,
				                  attn_drop: 0.0, ffd_drop: 0.0})
			logits_all.extend(logits.tolist())
			labels_all.extend(labels.tolist())
			train_loss_avg += loss_value_tr
			train_acc_avg += acc_tr
			train_acc_avg_final += acc_tr_final
			tr_step += 1
		train_nAUC = process.cal_nAUC(scores=np.array(logits_all), labels=np.array(labels_all))

		vl_step = 0
		vl_size = X_test.shape[0]
		val_acc_avg_final = 0
		logits_all = []
		labels_all = []
		loaded_graph = tf.get_default_graph()

		while vl_step * batch_size < vl_size:
			if (vl_step + 1) * batch_size > vl_size:
				break
			X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
			X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
			X_input_P = X_test_P[vl_step * batch_size:(vl_step + 1) * batch_size]
			X_input_P = X_input_P.reshape([-1, 10, 3])
			X_input_B = X_test_B[vl_step * batch_size:(vl_step + 1) * batch_size]
			X_input_B = X_input_B.reshape([-1, 5, 3])
			X_input_I = X_test_I[vl_step * batch_size:(vl_step + 1) * batch_size]
			X_input_I = X_input_I.reshape([-1, I_nodes, 3])
			loss_value_vl, acc_vl, acc_vl_final, logits, labels = sess.run([loss, accuracy, accuracy_final, aver_final_pred, lab_resh],
			                                 feed_dict={
				                                 J_in: X_input_J,
				                                 P_in: X_input_P,
				                                 B_in: X_input_B,
				                                 I_in: X_input_I,
				                                 J_bias_in: biases_J,
				                                 P_bias_in: biases_P,
				                                 B_bias_in: biases_B,
				                                 I_bias_in: biases_I,
				                                 lbl_in: y_test[vl_step * batch_size:(vl_step + 1) * batch_size],
				                                 # msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
				                                 is_train: False,
				                                 attn_drop: 0.0, ffd_drop: 0.0})
			logits_all.extend(logits.tolist())
			labels_all.extend(labels.tolist())
			val_loss_avg += loss_value_vl
			val_acc_avg += acc_vl
			val_acc_avg_final += acc_vl_final
			vl_step += 1
		val_nAUC = process.cal_nAUC(scores=np.array(logits_all), labels=np.array(labels_all))
		print('Training: loss = %.5f, acc = %.5f, nAUC = %.5f | Val: loss = %.5f, acc = %.5f, nAUC = %.5f' %
		      (train_loss_avg / tr_step, train_acc_avg_final / tr_step, train_nAUC,
		       val_loss_avg / vl_step, val_acc_avg_final / vl_step, val_nAUC))
		# print(att_w)

		if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
			# if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
			vnAUC_mx = val_nAUC
			if dataset == 'IAS':
				if val_acc_avg / vl_step > vacc_mx and epoch >= 10:
					vacc_early_model = val_acc_avg / vl_step
					vlss_early_model = val_loss_avg / vl_step
					checkpt_file = RN_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(
						time_step) + \
					               '-' + dataset + '/' + split \
					               + '_' + str(pre_epochs) + '_' + str(s_range) + '_' + str(
						pred_margin) + view_dir + '_' + str(FLAGS.bi) \
					               + '_' + str(round(vacc_early_model * 100, 1)) + '_' + str(
						round(vnAUC_mx * 100, 1)) + change + '.ckpt'
					print(checkpt_file)
					if save_flag == '1':
						saver.save(sess, checkpt_file)
			else:
				if val_acc_avg / vl_step > vacc_mx:
					vacc_early_model = val_acc_avg / vl_step
					vlss_early_model = val_loss_avg / vl_step
					# checkpt_file = RN_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(time_step) + \
					#                '-' + dataset + '/' + split \
					#                + '_' + str(pre_epochs) + '_' + str(s_range) + '_' + str(pred_margin) + view_dir + '_' + str(FLAGS.bi) \
					#                + '_' + str(round(vacc_early_model*100, 1)) + '_' + str(round(vnAUC_mx*100, 1)) + change + '.ckpt'
					checkpt_file = RN_dir + '_' + str(fusion_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(time_step) + \
					               '-' + dataset + '/' + split \
					               + '_' + str(pre_epochs) + '_' + str(s_range) + '_' + str(pred_margin) + view_dir + '_' + str(FLAGS.bi) \
					               + '_best' + change + '.ckpt'
					print(checkpt_file)
					if save_flag == '1':
						saver.save(sess, checkpt_file)
			vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
			vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
			curr_step = 0
		else:
			curr_step += 1
			if curr_step == patience:
				print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx, ', nAUC: ',  vnAUC_mx)
				print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model, ', nAUC: ', val_nAUC)
				break
		train_loss_avg = 0
		train_acc_avg = 0
		val_loss_avg = 0
		val_acc_avg = 0
	saver.restore(sess, checkpt_file)
	sess.close()

print('Dataset: ' + dataset)

print('----- Opt. hyperparams -----')
print('pre_train_epochs: ' + str(pre_epochs))
print('nhood: ' + str(nhood))
print('skeleton_nodes: ' + str(nb_nodes))
print('seqence_length: ' + str(time_step))
print('pretext: ' + str(pretext))
print('fusion_lambda: ' + str(fusion_lambda))
print('batch_size: ' + str(batch_size))
print('lr: ' + str(lr))
print('view: ' + FLAGS.view)
print('P: ' + FLAGS.P)
print('fusion_lambda: ' + FLAGS.fusion_lambda)
print('loss_type: ' + FLAGS.loss)
print('patience: ' + FLAGS.patience)
print('save_flag: ' + FLAGS.save_flag)

print('----- Archi. hyperparams -----')
print('structural relation matrix number: ' + str(Ps[0]))
print('LSTM_embed_num: ' + str(LSTM_embed))
print('LSTM_layer_num: ' + str(num_layers))

