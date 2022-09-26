import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

meta_path = 'experiments/runs/1596467701/checkpoints/model.meta'


with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('experiments/runs/1596467701/checkpoints/'))

    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('gemini_model.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())