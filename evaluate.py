from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import tensorflow as tf
import os, sys
from utils import process_i as process

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20
nhood = 1
ft_size = 3
time_step = 6

batch_size = 256
lr = 0.005  # learning rate

tf.app.flags.DEFINE_string('dataset', 'KS20', "Dataset: IAS, KS20 or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")
tf.app.flags.DEFINE_string('split', '', "for IAS-Lab testing splits (A or B)")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('model_dir', 'best', "model directory")  # 'best' will test the best model in current directory

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

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset
time_step = int(FLAGS.length)
split = FLAGS.split
model_dir = FLAGS.model_dir


def evaluate_reid(model_dir, dataset):
	if dataset == 'IAS':
		classes = list(range(11))
	elif dataset == 'KS20':
		classes = list(range(20))
	elif  dataset == 'KGBD':
		classes = list(range(164))
	checkpoint = model_dir + ".ckpt"
	print('Evaluating the model saved in ' + model_dir)
	loaded_graph = tf.get_default_graph()

	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpoint + '.meta')
		loader.restore(sess, checkpoint)
		# loader = tf.train.import_meta_graph(checkpt_file + '.meta')
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
		aver_pre = loaded_graph.get_tensor_by_name('Recognition/Recognition/add_30:0')
		accuracy = loaded_graph.get_tensor_by_name('Recognition/Recognition/Mean_4:0')
		loss = loaded_graph.get_tensor_by_name('Recognition/Recognition/Mean_5:0')
		rank_acc = {}
		en_to_pred = loaded_graph.get_tensor_by_name("Recognition/Recognition/StopGradient:0")

		X_train_J, X_train_P, X_train_B, X_train_I, y_train, X_test_J, X_test_P, X_test_B, X_test_I, y_test, \
		adj_J, biases_J, adj_P, biases_P, adj_B, biases_B, adj_I, biases_I, nb_classes = \
			process.gen_train_data(dataset=dataset, split=split, time_step=time_step,
			                       nb_nodes=nb_nodes, nhood=nhood, global_att=False, batch_size=batch_size, view='',
			                       reverse='0')
		# print(batch_size)
		X_train = X_train_J
		X_test = X_test_J
		vl_step = 0
		vl_size = X_test.shape[0]
		logits_all = []
		labels_all = []

		vl_step = 0
		vl_loss = 0.0
		vl_acc = 0.0
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
			y_input = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
			loss_value_vl, acc_vl, pred = sess.run([loss, accuracy, aver_pre],
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
					is_train: False,
					attn_drop: 0.0, ffd_drop: 0.0})
			for i in range(y_input.shape[0]):
				for K in range(1, len(classes) + 1):
					if K not in rank_acc.keys():
						rank_acc[K] = 0
					t = np.argpartition(pred[i], -K)[-K:]
					if np.argmax(y_input[i]) in t:
						rank_acc[K] += 1
			logits_all.extend(pred.tolist())
			labels_all.extend(y_input.tolist())
			vl_loss += loss_value_vl
			vl_acc += acc_vl
			vl_step += 1
		for K in rank_acc.keys():
			rank_acc[K] /= (vl_step * batch_size)
			rank_acc[K] = round(rank_acc[K], 4)
		val_nAUC = process.cal_nAUC(scores=np.array(logits_all), labels=np.array(labels_all))
		from sklearn.metrics import roc_curve, auc, confusion_matrix
		y_true = np.argmax(np.array(labels_all), axis=-1)
		y_pred = np.argmax(np.array(logits_all), axis=-1)
		print('\n### Re-ID Confusion Matrix: ')
		print(confusion_matrix(y_true, y_pred))
		print('### Rank-N Accuracy: ')
		print(rank_acc)
		print('### Test loss:', round(vl_loss / vl_step, 4), '; Test accuracy:', round(vl_acc / vl_step, 4),
		      '; Test nAUC:', round(val_nAUC, 4))
		exit()

if dataset == 'KS20':
	nb_nodes = 25
	I_nodes = 49
	batch_size = 64
elif dataset == 'IAS' or dataset == 'KGBD':
	nb_nodes = 20
	I_nodes = 39
elif dataset == 'CASIA_B':
	nb_nodes = 14
	I_nodes = 27
	batch_size = 128
if split == 'A':
	batch_size = 128
elif split == 'B':
	batch_size = 64
if dataset == 'KGBD':
	batch_size = 256


if model_dir == 'best':
	if dataset == 'IAS' and split == 'A':
		batch_size = 64
		model_dir = 'RN/IAS-A_59.4_86.7_formal'
	elif dataset == 'IAS' and split == 'B':
		model_dir = 'RN/IAS-B_69.8_90.4_formal'
	elif dataset == 'KS20':
		model_dir = 'RN/KS20_87.5_95.8_formal'
	elif dataset == 'KGBD':
		model_dir = 'RN/KGBD_99.5_99.6_formal'

evaluate_reid(model_dir, dataset)