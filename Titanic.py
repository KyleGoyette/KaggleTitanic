import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
def read_from_csv(filename_queue,test=False):
    reader=tf.TextLineReader(skip_header_lines=1)
    _, csv_row=reader.read(filename_queue)

    if test==False:
        record_defaults=[[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]

        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13=tf.decode_csv(csv_row,record_defaults=record_defaults)

        features=tf.pack([col2,col3,col4,col5,col6,col9,col10,col11,col12,col13])
        label=tf.pack([col7,col8])
        return features,label
    elif test==True:
        record_defaults=[[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]
        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11=tf.decode_csv(csv_row,record_defaults)
        features=tf.pack([col2,col3,col4,col5,col6,col7,col8,col9,col10,col11])
        return features




def input_pipe(batch_size,fname,num_epochs=None):
    filename=fname
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epochs,shuffle=True)
    example,label=read_from_csv(filename_queue)
    min_after_dequeue=150

    #example_batch,label_batch=tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
    return gen_batch(example,label,min_after_dequeue,batch_size)

def gen_batch(features,label,min_queue_examples,batch_size):
    num_preprocess_threads=1
    batch_features,batch_labels = tf.train.shuffle_batch([features,label],batch_size=batch_size,num_threads=num_preprocess_threads,capacity=min_queue_examples+3*batch_size,min_after_dequeue=min_queue_examples)
    return (batch_features,tf.reshape(batch_labels,[batch_size,2]))


def inference(inputs,eval=False):
    shape=inputs.get_shape()
    D=10
    #D=shape[1].value
    out_size1=30
    out_size2=10
    out_size3=4
    with tf.variable_scope('hidden1')as scope:
        if eval:
            scope.reuse_variables()
        W1=tf.get_variable('affine1',shape=[D,out_size1],initializer=tf.truncated_normal_initializer(stddev=0.3))
        b1=tf.get_variable('bias1',shape=[out_size1],initializer=tf.constant_initializer(0.0))
        hidden1=tf.nn.relu(tf.matmul(inputs,W1)+b1)
        loss1=tf.nn.l2_loss(W1)
    with tf.variable_scope('hidden2') as scope:
        if eval:
            scope.reuse_variables()
        W2=tf.get_variable('affine2',shape=[out_size1,out_size2],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2=tf.get_variable('bias2',shape=[out_size2],initializer=tf.constant_initializer(0.0))

        hidden2=tf.nn.relu(tf.matmul(hidden1,W2)+b2)
        loss2=tf.nn.l2_loss(W2)

    with tf.variable_scope('hidden3') as scope:
        if eval:
            scope.reuse_variables()
        W3=tf.get_variable('affine3',shape=[out_size2,2],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3=tf.get_variable('bias3',shape=[2],initializer=tf.constant_initializer(0.0))
        logits=tf.matmul(hidden2,W3)+b3
        loss3=tf.nn.l2_loss(W3)

    reg_loss=loss1+loss2+loss3
    return logits,reg_loss

def loss_calc(logits,labs,reg_loss,reg):

    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labs,name='trainloss'))

    loss+=reg*reg_loss
    tf.scalar_summary('losses',loss)
    return loss

def training(loss, lr,global_step):
  # Create the gradient descent optimizer with the given learning rate.

    lr=tf.train.exponential_decay(lr,global_step,100,0.96)
    tf.scalar_summary('learning_rate', lr)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads=optimizer.compute_gradients(loss)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Create a variable to track the global step.
    train_op = optimizer.minimize(loss,global_step=global_step)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return train_op

def evaluation(logits,labs):
    #labs=tf.argmax(labels,1)
    #correct=tf.nn.in_top_k(logits,labs,1)
    correct = tf.reduce_sum(tf.cast(tf.equal(logits,labs),tf.int32))
    return correct

def main(fname,batch_size,num_iters,reg,lr):


    global_step=tf.Variable(0,trainable=False)

    batch_features,batch_labels=input_pipe(batch_size,fname,num_epochs=(791*num_iters/batch_size)+2)
    norm_features = tf.nn.l2_normalize(batch_features,1)
    logits,l2_loss = inference(norm_features)

    labs = tf.argmax(batch_labels,1)
    loss = loss_calc(logits,labs,l2_loss,reg)
    preds=tf.argmax(logits,1)

    with tf.control_dependencies([labs,loss]):
        corr = evaluation(preds,labs)


    eval_features,eval_labels = input_pipe(100,'./datasets/train_crossval.csv',num_epochs=num_iters)
    norm_evals = tf.nn.l2_normalize(eval_features,1)
    logs, _ = inference(norm_evals,eval=True)
    eval_correct = evaluation(tf.argmax(logs,1),tf.argmax(eval_labels,1))


    train_op=training(loss,lr,global_step)

    #if not train:

    #    inputs=tf.placeholder(tf.float32,shape=None)
    #    logits,_=inference(inputs)
    #    pred=tf.argmax(logits)


    summary_op=tf.merge_all_summaries()
    saver=tf.train.Saver()
    coord=tf.train.Coordinator()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter('./logs', sess.graph)

        saver.restore(sess,'./models/TitanicModel_30_2-4990')

        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        acc_history=[]
        mean=0
        for iter in range(num_iters):
            if iter%100==1:
                _,total_loss,correct, eval_corr=sess.run([train_op,loss,corr,eval_correct])

                acc=correct/float(batch_size)
                acc_history.append(acc)
                eval_acc=np.sum(eval_corr)/float(100)
                print('Iteration: %d, Loss: %.2f Acc: %.2f Eval_acc: %.2f'%(iter,total_loss,acc,eval_acc))

            else:
                _,total_loss,correct=sess.run([train_op,loss,corr])


                acc=correct/float(batch_size)

                mean=(mean*(iter)+acc)/(iter+1)
                acc_history.append(mean)
                print('Iteration: %d, Loss: %.2f Acc: %.2f'%(iter,total_loss,acc))
            #print('Iteration: %d, Loss: %.2f Acc: %.2f Eval_acc: %.2f'%(iter,total_loss,acc,eval_acc))
            if iter%499==0:
                saver.save(sess,'./models/TitanicModel_30_3',global_step=iter)
            if (iter%10==1):
                summary_str=sess.run(summary_op)
                summary_writer.add_summary(summary_str,iter)

        plt.plot(range(0,num_iters),acc_history)
        plt.show()
        coord.request_stop()
        coord.join(threads)


def eval_feed():
    data=np.genfromtxt('./datasets/test_fixed.csv', delimiter=",")
    ids=data[1:,1]
    data=data[1:,2:]
    x=np.linalg.norm(data,axis=0)
    data=data/x

    return data,ids

def evaluate_tests():

    feature=tf.placeholder(dtype=tf.float32,shape=[1,10])

    logits,_=inference(feature)
    pred=tf.argmax(logits,1)

    saver=tf.train.Saver()

    with tf.Session() as sess:
        ckpt = saver.restore(sess,'./models/TitanicModel_30_3-4990')
        #if ckpt==None:
        #    print "NOPE"
        #sess.run([tf.initialize_all_variables()])
        feats,ids=eval_feed()

        final_predictions=[]
        for i in range(418):

            feed_dict={
                feature : np.expand_dims(feats[i,:],0)
            }
            prediction=sess.run([pred], feed_dict= feed_dict)
            prediction=np.sum(prediction)
            final_predictions.append(prediction)

    final_predictions1=[[ids[i].astype(int),final_predictions[i]] for i in range(0,418)]
    resultfile = open('output3.csv','wb')
    wr=csv.writer(resultfile,dialect='excel')
    wr.writerows(final_predictions1)

#main(fname='./datasets/train_fixed.csv',batch_size=350,num_iters=5000,reg=0.004,lr=0.0015)
evaluate_tests()
#print eval_feed()