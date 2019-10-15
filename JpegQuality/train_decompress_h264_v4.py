#uniform content loss + adaptive threshold + per_class_input + recursive G
#improvement upon cqf37
from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import pdb
import rawpy
import glob
import utils
from PIL import Image

#input_dir = '../Learning-to-See-in-the-Dark/dataset/SONY/Sony/short/'
#gt_dir = '../Learning-to-See-in-the-Dark/dataset/SONY/Sony/long/'
#checkpoint_dir = './result_srgb_v3/'
#result_dir = './result_srgb_v3/'
input_dir = './dataset/long/'
gt_dir    = './dataset/long/'
checkpoint_dir = './result_decomp_h264_v4b/'
result_dir = './result_decomp_h264_v4b/'

#get train and test IDs
train_fns = glob.glob(gt_dir + '00*_05*.JPG')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))
 
print(len(train_fns))
#jpg_v3_only   
#train_ids = []
#for i in range(0,58,1):
#    train_ids.append(i)


#os.system(


def compress_vpx(img, the_codec  , videorate):
    x = scipy.misc.imsave('temp.png',img)
    strcmd="ffmpeg -loglevel quiet -i temp.png -vcodec " + the_codec + " -b:v "+ str(videorate) + "k temp.webm;"   
    strcmd= strcmd +"rm temp.png;ffmpeg -loglevel quiet -i temp.webm temp.png;"   
    strcmd= strcmd + "rm temp.webm -f"
    
    
    print(strcmd)    
    r = os.popen(strcmd)

    #quit()
    x = scipy.misc.imread('temp.png')
    
    
    return x
    
    
def compress_avi(img, the_codec  , videorate):
    x = scipy.misc.imsave('temp.png',img)
    strcmd="ffmpeg -loglevel quiet  -i temp.png -vcodec " + the_codec + " -b:v " + str(videorate) + "k temp.avi;"   
    strcmd= strcmd+"rm temp.png;ffmpeg -loglevel quiet -i temp.avi temp.png;"   
    strcmd= strcmd + "rm temp.avi -f"
    print(strcmd)    
    r = os.popen(strcmd)
    #quit()
    x = scipy.misc.imread('temp.png')
    return x

def compress_h264(img, the_codec  , videorate):
    x = scipy.misc.imsave('temp.png',img)
    strcmd="ffmpeg -loglevel quiet  -i temp.png -vcodec " + the_codec + " -b:v " + str(videorate) + "k temp.mp4;"   
    strcmd= strcmd+"rm temp.png;ffmpeg -loglevel quiet -i temp.mp4 temp.png;"   
    strcmd= strcmd + "rm temp.mp4 -f"
    print(strcmd)    
    r = os.popen(strcmd)
    #quit()
    x = scipy.misc.imread('temp.png')
    return x

def compress2jpg(img,jpeg_quality):
    
    x = scipy.misc.imsave('temp.png',img)
    x = Image.open('temp.png').save('temp.jpeg','JPEG',quality = jpeg_quality)
    x = scipy.misc.imread('temp.jpeg')
    #print(jpeg_quality)
    return x

def compress2jpg_v2(fname, jpeg_quality=20):
    #x = scipy.misc.imsave('temp.png',img)
    x = Image.open(fname).save('temp.jpeg','JPEG',quality = jpeg_quality)
    x = scipy.misc.imread('temp.jpeg')
    return x

    
def gen_compress_input(x):
    #ecodec_index = np.random.randint(0,10)    
    ecodec_index = np.random.randint(0,2)    
    #only H264 works. not know why?
    videorate = 250 #for patchsize 512 
    #ecodec_index = 0
    if ecodec_index == 0: 
        x = compress2jpg(x, 5 + np.random.randint(0, 51))
    if ecodec_index == 1:
        x = compress_h264(x , "libx264" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index == 2:    
        x = compress_h264(x , "mpeg4" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index == 3:    
        x = compress_vpx(x , "vp8" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index == 4:    
        x = compress_vpx(x , "vp9" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index  == 5:
        x = compress_avi(x , "libxvid" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index  == 6:
        x = compress_avi(x , "msmpeg4" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index  == 7:
        x = compress_avi(x , "msmpeg4v2" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index  == 8:
        x = compress_avi(x , "mpeg2video" , videorate + np.random.randint(0, 20) * 25)
    if ecodec_index  == 9:
        x = compress_avi(x , "mpeg1video" , videorate + np.random.randint(0, 20) * 25)
    
    return x
        
patchsize = 1024 #patch size for training
ps= patchsize
save_freq = 100

DEBUG = 0
if DEBUG == 1:
  save_freq = 2
  train_ids = train_ids[0:5]
  test_ids = test_ids[0:5]



def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

#Change to RGB input
def network(input):
    # resnet method
    '''conv1=slim.conv2d(input,256,[3,3])
    x = conv1
    for i in range(32):
        x=utils.resBlock(x,256,scale=0.1)
    conv2=slim.conv2d(x,256,[3,3])
    x=conv1+conv2
    x = utils.upsample(x,2,256,None)
    out=x'''
    
    #512*512
    features = 32
    kernel_size_1 = 3
    conv1=slim.conv2d(input,features,[kernel_size_1,kernel_size_1], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,features,[kernel_size_1,kernel_size_1], rate=1, activation_fn=lrelu,scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    # 256*256 
    conv2=slim.conv2d(pool1,features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
    
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    
    
    # 128 * 128
    conv3=slim.conv2d(pool2,features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    # 64 * 64 
    conv4=slim.conv2d(pool3,features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    # 32 * 32
    conv5=slim.conv2d(pool4,features*16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,features*16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

    up6 =  upsample_and_concat( conv5, conv4, features*8, features*16  )
    # 64 * 64
    conv6=slim.conv2d(up6,  features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, features*4, features*8  )
    # 128 * 128
    conv7=slim.conv2d(up7,  features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, features*2, features*4 )

    # 256 * 256
    conv8=slim.conv2d(up8,  features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, features, features*2 )
    #512*512
    conv9=slim.conv2d(up9,  features,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,features,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')


    #conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    #out = tf.depth_to_space(conv10,2)
    conv10=slim.conv2d(conv9,3,[3,3], rate=1, activation_fn=None, scope='g_conv10')
    #conv11=slim.conv2d_in_plane(conv10, [11,11],activation_fn=None, scope='g_conv11')
    out = conv10
    return out

def network_notworking(input):

    #512*512
    features = 32
    kernel_size_1 = 3
    conv1=slim.conv2d(input,features,[kernel_size_1,kernel_size_1], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,features,[kernel_size_1,kernel_size_1], rate=1, activation_fn=lrelu,scope='g_conv1_2')
    if 0:
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

        # 256*256 
        conv2=slim.conv2d(pool1,features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv2=slim.conv2d(conv2,features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
    
    
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
        
        
        # 128 * 128
        conv3=slim.conv2d(pool2,features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3=slim.conv2d(conv3,features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

        # 64 * 64 
        conv4=slim.conv2d(pool3,features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4=slim.conv2d(conv4,features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

        # 32 * 32
        conv5=slim.conv2d(pool4,features*16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv5=slim.conv2d(conv5,features*16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

        up6 =  upsample_and_concat( conv5, conv4, features*8, features*16  )
        # 64 * 64
        conv6=slim.conv2d(up6,  features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6=slim.conv2d(conv6,features*8,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, features*4, features*8  )
        # 128 * 128
        conv7=slim.conv2d(up7,  features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7=slim.conv2d(conv7,features*4,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, features*2, features*4 )

        # 256 * 256
        conv8=slim.conv2d(up8,  features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8=slim.conv2d(conv8,features*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, features, features*2 )
    #512*512
    else:
        up9 = conv1

    conv9=slim.conv2d(up9,  features,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,features,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')


    #conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    #out = tf.depth_to_space(conv10,2)
    conv10=slim.conv2d(conv9,3,[3,3], rate=1, activation_fn=None, scope='g_conv10')
    #conv11=slim.conv2d_in_plane(conv10, [11,11],activation_fn=None, scope='g_conv11')
    out = conv10
    return out

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    #im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def pack_png(img):
    #pack Bayer image to 4 channels
    #im = raw.raw_image_visible.astype(np.float32) 
    #im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level
    im = np.float32(img) / 255.0 
    
    #im = np.expand_dims(im,axis=2) 
    #img_shape = im.shape
    #H = img_shape[0]
    #W = img_shape[1]

    #out = np.concatenate((im[0:H:2,0:W:2,:], 
    #                   im[0:H:2,1:W:2,:],
    #                   im[1:H:2,1:W:2,:],
    #                   im[1:H:2,0:W:2,:]), axis=2)
    out = im
    return out


sess=tf.Session()


#in_image=tf.placeholder(tf.float32,[None,patchsize,patchsize,4])
#gt_image=tf.placeholder(tf.float32,[None,patchsize*2,patchsize*2,3])
# PNG has same input out
in_image=tf.placeholder(tf.float32,[None,patchsize,patchsize,3])
gt_image=tf.placeholder(tf.float32,[None,patchsize,patchsize,3])

out_image=network(in_image)

print(gt_image)
print(out_image)

G_loss=tf.reduce_mean(tf.abs(out_image - gt_image)) 
#squared_loss =tf.squared_difference(out_image*255 , gt_image*255)
#G_loss=tf.reduce_mean(squared_loss)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

#Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['30.0'] = [None]*len(train_ids)
input_images['25.0'] = [None]*len(train_ids)
input_images['20.0'] = [None]*len(train_ids)
input_images['10.0'] = [None]*len(train_ids)
input_images['5.0'] = [None]*len(train_ids)
input_images['2.0'] = [None]*len(train_ids)
input_images['1.0'] = [None]*len(train_ids)
input_images['0.5'] = [None]*len(train_ids)
input_images['0.25'] = [None]*len(train_ids)
input_images['0.2'] = [None]*len(train_ids)
input_images['0.1'] = [None]*len(train_ids)
input_images['0.05'] = [None]*len(train_ids)
input_images['0.017'] = [None]*len(train_ids)
input_images['0.333'] = [None]*len(train_ids)
input_images['0.033'] = [None]*len(train_ids)
input_images['0.077'] = [None]*len(train_ids)
input_images['0.04'] = [None]*len(train_ids)
input_images['0.004'] = [None]*len(train_ids)
input_images['0.008'] = [None]*len(train_ids)


g_loss = np.zeros((5000,1))



allfolders = glob.glob(result_dir + '0*00')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-5:]))
print("lastepoch: ", lastepoch)
learning_rate = 1e-4

f = open('t.log', 'w')
f.write("...\n") 
f.close()
prev_time = time.time()

batch_size = 1

st=time.time()

border_size = 500 
cmaxlip=1.0

def getBatches(indices, epoch):
    input_patches = []
    gt_patches = []
    ratios = []    
    gt_exposures = []
    for ind in indices:
        train_id = train_ids[ind]        
         
        #in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        #READ PNG file
        #print('%05d_00*.PNG'%train_id)
        in_files = glob.glob(input_dir + '%05d_00*_0*.JPG'%train_id)
        
        random_i = np.random.randint(0,len(in_files))          
        in_path = in_files[random_i]
        _, in_fn = os.path.split(in_path)
        in_exposure =  float(in_fn[9:-5])                  
        
        st=time.time()
        
        if input_images[str(in_exposure)][ind] is None:
            im = scipy.misc.imread( in_path)                              
            im = np.expand_dims(im[border_size:-border_size, border_size:-border_size, :],axis = 0)
            input_images[str(in_exposure)][ind] = im

        #crop
        H = input_images[str(in_exposure)][ind].shape[1]
        W = input_images[str(in_exposure)][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        #source/Dest has same patch size
        gt_patch = input_images[str(in_exposure)][ind][:,yy:yy+ps,xx:xx+ps,:]       
        #if epoch >14100:
        #    input_patch = compress2jpg(gt_patch[0], 2 + np.random.randint(0, 42)*2)
        #else:
        #    input_patch = compress2jpg(gt_patch[0], 5 + np.random.randint(0, 19))
        #input_patch = np.expand_dims(input_patch, axis = 0)
        #
        
        input_patch = gen_compress_input(gt_patch[0])
        input_patch = np.expand_dims(input_patch, axis = 0)
        
        input_patch = np.float32(input_patch/255.0)
        gt_patch    = np.float32(gt_patch/255.0)
       
        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
        input_patches.append(input_patch[0])
        gt_patches.append(gt_patch[0])
        ratio=1.0
        ratios.append(ratio)
        gt_exposure = 10.0
        gt_exposures.append(gt_exposure)
        
        #print(input_patch.shape, gt_patch.shape)
    return input_patches, gt_patches, ratios, gt_exposures
#lastepoch =4000 
#lastepoch = 1500

#lastepoch = 0
for epoch in range(lastepoch,3001):
    #if os.path.isdir(result_dir + "/%05d"%epoch):
    #    continue    
    cnt=0
    #if epoch > 2000:
    #    learning_rate = 1e-5
        
    if epoch > 500:
        learning_rate = 1e-5
             
    print ("Epoches: ", epoch, time.time()- prev_time)
    f = open(result_dir + "/t%05d.log"%epoch, 'w')
    prev_time = time.time()
    
    indices = []
    indcnt = 0
    for ind in np.random.permutation(len(train_ids)):
   
        indices.append(ind)
        indcnt =  indcnt+1
        if indcnt%batch_size  == 0:        
            
            input_patches, gt_patches, ratios, gt_exposures =  getBatches(indices, epoch)
            _,G_current,output=sess.run([G_opt,G_loss,out_image],feed_dict={in_image:input_patches,gt_image:gt_patches,lr:learning_rate})
            output = np.minimum(np.maximum(output,0),1)
            g_loss[ind]=G_current
    
            #print("%d %d Loss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st))
            f.write(("%d %d %d %d Loss=%.3f Time=%.3f\n"%(epoch,ind,ratios[0], gt_exposures[0],  np.mean(g_loss[np.where(g_loss)]),time.time()-st)))
    
            if epoch%save_freq==0:
                if not os.path.isdir(result_dir + '%05d'%epoch):
                    os.makedirs(result_dir + '%05d'%epoch)
                for ind2 in range(0,batch_size):
                    train_id = indices[ind2]
                    train_id = train_ids[train_id]
                    ratio = ratios[ind2]
                    temp = np.concatenate((input_patches[ind2][:,:,:],gt_patches[ind2][:,:,:],output[ind2][:,:,:]),axis=1) 
                    print(result_dir + '%05d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))    
                    scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%05d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))
                    
            #reset
            indcnt = 0
            indices = []
            #saver.save(sess, checkpoint_dir + 'model.ckpt')
            

    saver.save(sess, checkpoint_dir + 'model.ckpt')
    f.close()
