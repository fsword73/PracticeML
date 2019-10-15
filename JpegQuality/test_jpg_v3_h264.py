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
from PIL import Image

# --validation 1 --test_dir ./dataset/long/  --out_dir ./test_output_decompress_h264_jpg5/ --checkpoint_dir ./result_decomp_h264_v3/ --raw_type 0 --jpg_quality 5


# --validation 1 --test_dir ./dataset/vrate/  --out_dir ./test_output_decompress_h264/ --checkpoint_dir ./result_jpg_v3_decomp/ --raw_type 0 --ratio 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", default='./test/')
parser.add_argument("--out_dir", default='./output/')
parser.add_argument("--checkpoint_dir", default='./checkpoint/Sony')
parser.add_argument("--ratio", type=int, default=100)
parser.add_argument("--raw_type", type=int, default=0)
parser.add_argument("--validation",type=int, default=0)   
parser.add_argument("--blacklevel", type=int, default=535)
parser.add_argument("--gt_dir", default='./dataset/long/')
parser.add_argument("--jpg_quality", type=int, default=0)



args = parser.parse_args()


test_dir = args.test_dir
checkpoint_dir = args.checkpoint_dir
result_dir = args.out_dir
gt_dir =  args.gt_dir
raw_type = args.raw_type
ratio = args.ratio
blackLevel = args.blacklevel

from scipy.ndimage.interpolation import zoom


#PatchSize for inference 
patchsize = 1024 #patch size for training
ps= patchsize



def psnr(im1,im2):
    diff = numpy.abs(im1 - im2)
    rmse = numpy.sqrt(diff).sum()
    psnr = 20*numpy.log10(255/rmse)
    return psnr
    

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
def network_orig(input):
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

def network(input):

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
#Network 
sess=tf.Session()


#in_image=tf.placeholder(tf.float32,[None,patchsize,patchsize,4])
#gt_image=tf.placeholder(tf.float32,[None,patchsize*2,patchsize*2,3])
# PNG has same input out
in_image=tf.placeholder(tf.float32,[None,patchsize,patchsize,3])
gt_image=tf.placeholder(tf.float32,[None,patchsize,patchsize,3])

out_image=network(in_image)

#No Optimizer for Infernece Only

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    
#
def compress2jpg(img,jpeg_quality):

    x = scipy.misc.imsave('temp.png',img)
    x = Image.open('temp.png').save('temp.jpeg','JPEG',quality = jpeg_quality)
    x = scipy.misc.imread('temp.jpeg')
    #print(jpeg_quality)
    return x

cmaxlip=100.0
#cmaxlip=1.0  
def Decompress_Validation_test():
    #find all 1/30 files from test_dir 
    
    print (test_dir + "*_00_0*")
    if raw_type == 0:
        test_files =  glob.glob(test_dir + "*_00_0*")
    else:
        test_files =  glob.glob(test_dir + "*_00_0*") 

    
        
    print(len(test_files))
    #Inference for each file
    for i in range(len(test_files)):
        in_path = test_files[i]     
        _, in_fn = os.path.split(in_path)        
        
     
        #print(test_files[i]) 
        #In/Out Exposure
        #in_exposure = 0.033
        in_exposure =  1
        gt_exposure =  1
        
        #Jian's Cannon only:  ratio =  min((int)(gt_exposure/in_exposure),300)/8
        
        ratio =  1
        
        if(raw_type == 1):
            in_im = rawpy.imread(in_path)
            in_im = in_im.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            in_im = np.expand_dims(in_im,axis = 0)
            
            
            gt_im = np.float32(in_im/65536.0)* 255
            
            in_im = compress2jpg(in_im[0], args.jpg_quality)
            in_im = np.expand_dims(in_im,axis = 0)
            in_im = np.float32(in_im/255.0)            
            
        else:
            in_im = scipy.misc.imread( in_path)   
            gt_im = np.float32(in_im/255.0)
            #gt_im = np.expand_dims(gt_im,axis = 0)
            
            in_im = compress2jpg(in_im, args.jpg_quality)            
            scipy.misc.toimage(in_im,  high=255, low=0, cmin=0, cmax=255).save(result_dir + "compress/"+ in_fn )
            in_im = np.float32(in_im /255.0)            
            in_im = np.expand_dims(in_im,axis = 0)
        

        #alignment to patchsize
        #print(in_im.shape)
        H = in_im.shape[1] 
        W = in_im.shape[2]
                
        
        H2 = int ((H + patchsize-1)/patchsize)
        H2 = H2 *  patchsize
        
        W2 = int ((W + patchsize-1)/patchsize)
        W2 = W2 *  patchsize
        
        
        
        output_img = np.zeros([H, W, 3])
        
        #force to 512x512
        input_img = np.zeros([1, H2, W2, 3])        
        in_im[:,:,0] = in_im[:,:,0] * 0.6
        #in_im[:,:,2] = in_im[:,:,2] * 0.8
        input_img[0, 0:H, 0:W,:] = in_im
         
        for yy in range(0, H2, patchsize):
            for xx in range(0, W2, patchsize):    
              y_max = yy + patchsize
              x_max = xx + patchsize              
               
              input_patch = np.minimum(input_img[0,yy:y_max,xx:x_max,:], cmaxlip)
              input_patch = np.expand_dims(input_patch,axis = 0)                            
              output_patch =sess.run(out_image,feed_dict={in_image: input_patch})              
              output_patch = np.minimum(np.maximum(output_patch,0),1)              
              output_patch = output_patch[0,:,:,:]       
              
              x_max = min(x_max, W)
              y_max = min(y_max, H)
              
              output_img[yy:y_max,xx:x_max,:] = output_patch[0:y_max-yy, 0:x_max-xx,:]
              
        #ouput CNN output and compute PSNR and SSIM     
                  
        scipy.misc.toimage(output_img*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + in_fn )
        from skimage import  measure
        psnr = measure.compare_psnr(gt_im, output_img)
        ssim = measure.compare_ssim(gt_im, output_img, multichannel=True)
        print( in_fn + ",psnr,%.2f,ssim,%.4f" % (psnr, ssim)) 
        
def Validation_test():
    #find all 1/30 files from test_dir 
    
    print (test_dir + "*_00_0.*")
    if raw_type == 0:
        test_files =  glob.glob(test_dir + "*_00_0.*")
    else:
        test_files =  glob.glob(test_dir + "*_00_0.*") 

    
        
    print(len(test_files))
    #Inference for each file
    for i in range(len(test_files)):
        in_path = test_files[i]        
     
        #print(test_files[i]) 
        _, in_fn = os.path.split(in_path)
        
        train_id = in_fn[0:8]
        
        #find gt_file 
        gt_paths = glob.glob(gt_dir + train_id + "05s.*")
        if(len(gt_paths) == 0):
            gt_paths = glob.glob(gt_dir + train_id + "*s.*")
        for gt_ind in range(len(gt_paths)):
            gt_path = gt_paths[gt_ind]
            _, gt_fn = os.path.split(gt_path)
            gt_exposure =  float(gt_fn[9:-5])
            
            if(gt_exposure < 10):
                break;
            
        
        print("gt_path:" + gt_path)
        #In/Out Exposure
        #in_exposure = 0.033
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5]) 
        
        #Jian's Cannon only:  ratio =  min((int)(gt_exposure/in_exposure),300)/8
        
        ratio =  min((int)(gt_exposure/in_exposure),300)
        
        if(raw_type == 1):
            in_im = rawpy.imread(in_path)
            in_im = in_im.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            in_im = np.expand_dims(in_im,axis = 0)
            
            gt_im = rawpy.imread(gt_path)
            gt_im = gt_im.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            
            
            in_im = np.float32(in_im/65536.0) * ratio
            gt_im    = np.float32(gt_im/65536.0) 
            
        else:
            in_im = scipy.misc.imread( in_path)
            in_im = np.expand_dims(in_im,axis = 0)
            
            gt_im = scipy.misc.imread( gt_path)            

            in_im = np.float32(in_im/255) * ratio
            gt_im    = np.float32(gt_im/255) 
        

        #alignment to patchsize
        #print(in_im.shape)
        H = in_im.shape[1] 
        W = in_im.shape[2]
                
        
        H2 = int ((H + patchsize-1)/patchsize)
        H2 = H2 *  patchsize
        
        W2 = int ((W + patchsize-1)/patchsize)
        W2 = W2 *  patchsize
        
        
        
        output_img = np.zeros([H, W, 3])
        
        #force to 512x512
        input_img = np.zeros([1, H2, W2, 3])        
        in_im[:,:,0] = in_im[:,:,0] * 0.6
        #in_im[:,:,2] = in_im[:,:,2] * 0.8
        input_img[0, 0:H, 0:W,:] = in_im
         
        for yy in range(0, H2, patchsize):
            for xx in range(0, W2, patchsize):    
              y_max = yy + patchsize
              x_max = xx + patchsize              
               
              input_patch = np.minimum(input_img[0,yy:y_max,xx:x_max,:], cmaxlip)
              input_patch = np.expand_dims(input_patch,axis = 0)                            
              output_patch =sess.run(out_image,feed_dict={in_image: input_patch})              
              output_patch = np.minimum(np.maximum(output_patch,0),1)              
              output_patch = output_patch[0,:,:,:]       
              
              x_max = min(x_max, W)
              y_max = min(y_max, H)
              
              output_img[yy:y_max,xx:x_max,:] = output_patch[0:y_max-yy, 0:x_max-xx,:]
              
        #ouput CNN output and compute PSNR and SSIM     
                  
        scipy.misc.toimage(output_img*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + train_id + "_00_"+ in_fn[9:-5]+ ".JPG" )
        from skimage import  measure
        psnr = measure.compare_psnr(gt_im, output_img)
        ssim = measure.compare_ssim(gt_im, output_img, multichannel=True)
        print( in_fn + ",psnr,%.2f,ssim,%.4f" % (psnr, ssim)) 
        
def Inference_test():
    #find all 1/30 files from test_dir 
    print (test_dir + "*.*")
    if raw_type == 0:
        test_files =  glob.glob(test_dir + "*.*")
    else:
        test_files =  glob.glob(test_dir + "*.*") 

    #Inference for each file
    print(len(test_files))
    
    for i in range(len(test_files)):
        in_path = test_files[i]
        _, in_fn = os.path.split(in_path)
        
        ratio =  args.ratio
        
        if(raw_type == 1):
            in_im = rawpy.imread(in_path)
            in_im = in_im.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            in_im = np.expand_dims(in_im,axis = 0)
            
            in_im = np.float32(in_im/65536.0) * ratio
            
        else:
            in_im = scipy.misc.imread( in_path)
            in_im = np.expand_dims(in_im,axis = 0)
            
            in_im = np.float32(in_im/255) * ratio
        

        #alignment to patchsize 
        #print(in_im.shape)
        H = in_im.shape[1] 
        W = in_im.shape[2]
                
        
        H2 = int ((H + patchsize-1)/patchsize)
        H2 = H2 *  patchsize
        
        W2 = int ((W + patchsize-1)/patchsize)
        W2 = W2 *  patchsize
        
        
        
        output_img = np.zeros([H, W, 3])
        
        #force to 512x512
        input_img = np.zeros([1, H2, W2, 3])        
        input_img[0, 0:H, 0:W,:] =in_im
         
        for yy in range(0, H2, patchsize):
            for xx in range(0, W2, patchsize):    
              y_max = yy + patchsize
              x_max = xx + patchsize              
               
              input_patch = np.minimum(input_img[0,yy:y_max,xx:x_max,:], cmaxlip)
              input_patch = np.expand_dims(input_patch,axis = 0)                            
              output_patch =sess.run(out_image,feed_dict={in_image: input_patch})              
              output_patch = np.minimum(np.maximum(output_patch,0),1)              
              output_patch = output_patch[0,:,:,:]       
              
              x_max = min(x_max, W)
              y_max = min(y_max, H)
              
              output_img[yy:y_max,xx:x_max,:] = output_patch[0:y_max-yy, 0:x_max-xx,:]
              
        #ouput CNN output and compute PSNR and SSIM              
        scipy.misc.toimage(output_img*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + in_fn[0:-4] + ".JPG" )
        print( in_fn + "\n")
                         
            

if(args.validation > 0):
    if args.jpg_quality > 0:
        Decompress_Validation_test()
    else:
        Validation_test()
else:
    Inference_test()     
