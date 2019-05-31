import tensorflow as tf
from data_utils import Data
from char_cnn import CharConvNet
import numpy as np

if __name__ == '__main__':
    with open('config.py', 'r') as source_file:
        exec(source_file.read())
    sess = tf.Session()

    # print "Loading data ....",
    train_data = Data(data_source=config.train_data_source,
                      alphabet=config.alphabet,
                      l0=config.l0,
                      batch_size=config.batch_size,
                      no_of_classes=config.no_of_classes)
    train_data.loadData()
    dev_data = Data(data_source=config.dev_data_source,
                    alphabet=config.alphabet,
                    l0=config.l0,
                    batch_size=config.batch_size,
                    no_of_classes=config.no_of_classes)

    dev_data.loadData()

    num_batches_per_epoch = int(train_data.getLength() / config.batch_size) + 1
    num_batch_dev = dev_data.getLength()
    # print "Loaded"
    #

   #print(dev_data.data)


    saver = tf.train.import_meta_graph('./runs/1518482468/checkpoints/model-378900.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./runs/1518482468/checkpoints/'))

    graph = tf.get_default_graph()
    # Accessing Variables
    input_x = graph.get_tensor_by_name("Input-Layer/input_x:0")
    input_y = graph.get_tensor_by_name("Input-Layer/input_y:0")
    dropout_keep_prob = graph.get_tensor_by_name("Input-Layer/dropout_keep_prob:0")
    #print(graph.get_operations())
    #Accessing oploss
    predictions = graph.get_tensor_by_name("OutputLayer/scores:0")
    pred_list  = []
    for k in range(num_batches_per_epoch):
        batch_x, batch_y = train_data.getBatchToIndices(k, shuffle=False)
        #print(batch_x[22])
        #print(batch_y[22])
        #print(train_data.getBatch(0,shuffle=False))
        #print('----------------------')

        preds = sess.run(predictions, feed_dict={input_x:batch_x, input_y:np.random.randint(0,1,size=(128,12)),
                                               dropout_keep_prob:1.0})
        output = (np.argmax(preds, 1) + 1 ) % 12
        ous = output.tolist()
        #print(ous)
        pred_list.extend(ous)
    print(pred_list)
    print(len(pred_list))