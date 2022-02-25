# Fixing paths
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import py_function as py_func
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import r2_score

from smartsim.ml.tf import freeze_model

np.random.seed(10)
tf.random.set_seed(10)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)


# Custom metric
def r2_2(y_true, y_pred):
    res = py_func(r2_score, [y_true, y_pred], tf.float64)
    return res


# Data loader
def load_data():
    # Load data
    total_data = np.load(HERE+'/Total_dataset.npy')

    total_samples = np.shape(total_data)[0]
    np.random.shuffle(total_data) # Randomizing data

    # Scaling
    # preproc = Pipeline([('minmaxscaler', MinMaxScaler()),('stdscaler', StandardScaler())])
    # total_data = preproc.fit_transform(total_data[:,:])

    mv_scaler = StandardScaler()
    mv_scaler.fit(total_data[:,:])

    mv_scaler_filename = "mv_scaler.save"
    joblib.dump(mv_scaler, mv_scaler_filename)
    total_data[:,:] = mv_scaler.transform(total_data[:,:])

    # Randomize the data
    np.random.shuffle(total_data)

    # Some shapes
    num_outputs = 1
    num_inputs = np.shape(total_data)[1] - num_outputs

    # Inputs and outputs
    training_inputs = total_data[:,0:num_inputs]
    training_outputs = total_data[:,num_inputs:].reshape(total_samples,num_outputs) # Just to make sure

    return num_inputs, training_inputs, num_outputs, training_outputs


def get_model(num_inputs,num_outputs,num_layers,num_neurons):
    # Define model architecture here
    ph_input = keras.Input(shape=(num_inputs,),name='input_placeholder')
    hidden_layer = keras.layers.Dense(num_neurons,activation='tanh')(ph_input)

    for layer in range(num_layers):
        hidden_layer = keras.layers.Dense(num_neurons,activation='tanh')(hidden_layer)

    output = keras.layers.Dense(num_outputs,activation='linear',name='output_value')(hidden_layer)

    model = keras.Model(inputs=[ph_input],outputs=[output])
    #keras.utils.plot_model(model, 'ml_model.png', show_shapes=True)
    my_adam = keras.optimizers.Adam(lr=0.001, decay=0.0)

    model.compile(optimizer=my_adam,loss={'output_value': 'mean_squared_error'},metrics=[r2_2])
    model.summary()

    return model


def fit_model(training_inputs,training_outputs,model,num_epochs):
    model_path = './model.h5'
    # Callbacks
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False)
    csv_logger = keras.callbacks.CSVLogger('training.log')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    history = model.fit({'input_placeholder': training_inputs},
                        {'output_value': training_outputs},
                        epochs=num_epochs,
                        batch_size=1024,callbacks=[tbCallBack,checkpoint,csv_logger,earlystopping],
                        validation_split=0.1,verbose=1)

    # Get the best model from validation loss
    loaded_model = keras.models.load_model(model_path,custom_objects={'r2_2': r2_2})

    model_path, inputs, outputs = freeze_model(loaded_model, os.getcwd(), "ML_SA_CG.pb")

    graph = load_graph('ML_SA_CG.pb')
    for op in graph.get_operations():
        print(op.name)

    input_tensor = model.inputs[0]
    output_tensor = model.outputs[0]
    print("Inputs: "+str(input_tensor))
    print("Outputs: "+str(output_tensor))


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.compat.v2.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # # Some shell commands to clean the directory
    # os.system('rm model.h5')
    # os.system('rm mv_scaler.save')
    # os.system('rm -rf Graph')

    num_inputs, training_inputs, num_outputs, training_outputs = load_data()
    model = get_model(num_inputs,num_outputs,6,40)
    fit_model(training_inputs,training_outputs,model,500)
