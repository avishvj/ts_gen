from __future__ import print_function
import os, sys, time, random
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from rdkit import Chem, Geometry
from optparse import OptionParser

# mlflow
import mlflow
import mlflow.sklearn

# syspath for current directory
sys.path.insert(0, "../model/")
# from util import render_pymol
from G2C import G2C

# constants
MAX_NUM_BONDS = 10.

def eval_metrics(actual, pred):
        # this is the one in G2C
        return

def prepare_batch(batch_mols, max_size, elements):
        """ Returns batch with atom features, edge features, and coordinates initialised.
            Edge features based on topological distances (number of bonds between atoms) and 3D RBF distances. 
        """

        # func constants
        num_elements = len(elements)

        # initialise
        size = len(batch_mols)
        V = np.zeros((size, max_size, num_elements + 1), dtype = np.float32)
        E = np.zeros((size, max_size, max_size, 3), dtype = np.float32)
        sizes = np.zeros(size, dtype = np.int32)
        coordinates = np.zeros((size, max_size, 3), dtype = np.float32)

        # build atom features
        for bx in range(size):
        reactant, ts, product = batch_mols[bx]
        num_atoms = reactant.GetNumAtoms()
        sizes[bx] = int(num_atoms)

        # topological distances matrix
        D = (Chem.GetDistanceMatrix(reactant) + Chem.GetDistanceMatrix(product)) / 2
        D[D > MAX_NUM_BONDS] = MAX_NUM_BONDS

        # 3D rbf matrix
        D_3D_rbf = np.exp(-((Chem.Get3DDistanceMatrix(reactant) + Chem.Get3DDistanceMatrix(product)) / 2)) 

        for i in range(num_atoms):

                # edge features (stays bonded and bond aromatic?, stays bonded?, 3D rbf distance)
                for j in range(num_atoms):
                        # if stays bonded
                        if D[i][j] == 1.: 
                                # if aromatic bond
                                if reactant.GetBondBetweenAtoms(i, j).GetIsAromatic():
                                        E[bx, i, j, 0] = 1.
                                E[bx, i, j, 1] = 1
                        # add 3D rbf dist
                        E[bx, i, j, 2] = D_3D_rbf[i][j]

                # node features
                atom = reactant.GetAtomWithIdx(i)
                e_ix = elements.index(atom.GetSymbol())
                V[bx, i, e_ix] = 1.
                V[bx, i, num_elements] = atom.GetAtomicNum() / 10.

                # recover coordinates
                pos = ts.GetConformer().GetAtomPosition(i)
                np.asarray([pos.x, pos.y, pos.z])
                coordinates[bx, i, :] = np.asarray([pos.x, pos.y, pos.z])
        
        batch_dict = {"nodes": tf.constant(V), "edges": tf.constant(E), "sizes": tf.constant(sizes), "coordinates": tf.constant(coordinates)}

        return batch_dict



if __name__ == "__main__":

        # read in args
        parser = OptionParser()
        
        # have relevant parsing options
        parser.add_option("--restore", dest="restore", default=None)
        parser.add_option("--layers", dest="layers", default=2)
        parser.add_option("--hidden_size", dest="hidden_size", default=128)
        parser.add_option("--iterations", dest="iterations", default=3)
        parser.add_option("--batch_size", dest="batch_size", default=8)
        parser.add_option("--epochs", dest="epochs", default=20)
        
        args, _ = parser.parse_args()

        # NN params
        layers = int(args.layers)
        hidden_size = int(args.hidden_size)
        iterations = int(args.iterations)

        # training params
        BATCH_SIZE = int(args.batch_size)
        EPOCHS = int(args.epochs)
        best_val_loss = 9e99

        # load data (don't remove hydrogens and don't sanitise) TODO? functionise this
        reactant_file = "../create_figs/model_data/train_reactants.sdf"
        train_r = Chem.SDMolSupplier(reactant_file, removeHs=False, sanitize=False)
        train_r = [r for r in train_r]        
        
        ts_file = "../create_figs/model_data/train_ts.sdf"
        train_ts = Chem.SDMolSupplier(ts_file, removeHs=False, sanitize=False)
        train_ts = [ts for ts in train_ts]        
        
        product_file = "../create_figs/model_data/train_products.sdf"
        train_p = Chem.SDMolSupplier(product_file, removeHs=False, sanitize=False)
        train_p = [p for p in train_p]        

        train_data = list(zip(train_r, train_ts, train_p))

        # dataset dimensions
        elements = "HCNO"
        num_elements = len(elements)
        max_size = max([x.GetNumAtoms() for x,y,z in data]) # TODO? max for xyz

        num_train = len(train_data)
        num_valid = int(round(num_data / 8))

        # train:val splits
        data_train = train_data[ :num_train - num_valid]
        data_val = train_data[num_train - num_valid: ]

        # prepare train:val batches and iterators
        with tf.variable_scope("Dataset"):
                dtypes = [tf.float32, tf.float32, tf.int32, tf.float32]
                names = ['nodes', 'edges', 'sizes', 'coordinates']
                shapes = [[BATCH_SIZE, max_size, num_elements + 1], [BATCH_SIZE, max_size, max_size, 3], [BATCH_SIZE], [BATCH_SIZE, max_size, 3]]
                number_of_threads_train_n = 3

                ds_train = tf.data.Dataset.from_tensor_slices(prepare_batch(data_train)).cache().batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
                ds_valid = tf.data.Dataset.from_tensor_slices(prepare_batch(data_valid)).cache().batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

                iterator = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
                next_element = iterator.get_next()

                training_init_op = iterator.make_initializer(ds_train)
                validation_init_op = iterator.make_initializer(ds_valid)

        # mlflow: run model, fit train data, predict on val and get metrics out
        # also get some mlflow log metrics  
        # build data, then mlflow
        # for plots: D_init, W, GNN embeddings 
        # for model training/plots: best_model.ckpt, last_model.ckpt, HP values
        # for 
        # 

        # tensorflow inside mlflow scope?
        #       - dataset
        #       - build model
        #       - launch session and do
        #       - print and/or save metrics + best models
        #       - save D_init, W, GNN embeddings
        with mlflow.start_run():

                # build model
                dgnn = G2C(max_size=max_size, node_features=num_elements + 1, edge_features=3, layers=layers, hidden_size=hidden_size, iterations=iterations, input_data=next_element)

                # look at total parameters possible
                total_parameters = 0
                for variable in tf.trainable_variables():
                        shape = variable.get_shape()
                        variable_parameters = 1
                        for dim in shape:
                                variable_parameters *= dim.value
                        total_parameters += variable_parameters
                print('Total trainable variables: {}'.format(total_parameters))

                # launch session
                config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

                with tf.Session(config=config) as sess:
                        
                        # initialisation
                        print("Initialising...")
                        init = tf.global_variables_initializer()
                        sess.run(init)

                        # build batch summaries
                        #summary_op = tf.summary.merge_all()
                        #summary_writer = tf.summary.FileWriter(base_folder, sess.graph)
                        # how to use mlflow here instead?

                        # variable saving
                        saver = tf.train.Saver()
                        if args.restore is not None:
                                saver.restore(sess, args.restore)
                        print("TensorFlow set up complete.")

                        counter = 0
                        for epoch in range(EPOCHS):
                                sess.run(training_init_op)
                                batches_trained = 0

                                try:
                                        while True:
                                                _, _, summ = sess.run(
                                                [dgnn.train_op, dgnn.debug_op, summary_op])
                                                # summary_writer.add_summary(summ, counter) MLflow here!
                                                batches_trained += 1
                                                counter += 1
                                                print("Batches trained: ", batches_trained)
                                except tf.errors.OutOfRangeError as e:
                                        pass
                                sess.run(validation_init_op)

                                X = np.empty([len(data_valid), max_size, 3])
                                valid_loss_all = np.empty([len(data_valid), max_size, max_size])
                                D_mask = np.empty([len(data_valid), max_size, max_size])
                                batches_validated = 0

                                try:
                                        while True:
                                                _, X[batches_validated * BATCH_SIZE:(batches_validated + 1) * BATCH_SIZE, :, :], \
                                                valid_loss_all[batches_validated * BATCH_SIZE:(batches_validated + 1) * BATCH_SIZE, :, :], \
                                                D_mask[batches_validated * BATCH_SIZE:(batches_validated + 1) * BATCH_SIZE, :, :] = sess.run([dgnn.debug_op, dgnn.tensors["X"], dgnn.loss_distance_all, dgnn.masks["D"]])
                                                batches_validated += 1
                                                print("Batches validated: ", batches_validated)
                                except tf.errors.OutOfRangeError as e:
                                        pass
                
                                losses = [np.sum(valid_loss_all[i] * D_mask[i]) / np.sum(D_mask[i]) for i in range(X.shape[0])]
                                val_loss = np.mean(np.asarray(losses))
                                save_path_last = saver.save(sess, base_folder + "last_model.ckpt")

                                if val_loss < best_val_loss:
                                        best_val_loss = val_loss
                                        save_path = saver.save(sess, base_folder + "best_model.ckpt")

                                print("Validation Loss: {}".format(val_loss))
