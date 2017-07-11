# spatial-IRNN   

This is the core implementation for the following publication.

SIAMESE RECURRENT ARCHITECTURE FOR VISUAL TRACKING

Version 1.0, Copyright(c) July, 2017

Xiaqing Xu, Bingpeng Ma, Hong Chang, Xilin Chen.

All Rights Reserved.

## Installation
Requires a recent version of caffe.

Then, simply copy the files in include/ and src/ to their corresponding directories.

Attention! Please replace the 'filler.hpp' of original caffe with the one in this folder.
A new 'IdentityFiller' is implemented in our file.

### Patching the proto file
You need to merge the proto buffer definition in patch.proto with src/caffe/proto/caffe.proto.

### Adding the layer for permuting
For the efficiency of computing, the spatial-IRNN needs to permute the input blob's shape first.
We use the 'permute layer' of SSD <https://github.com/weiliu89/caffe/tree/ssd> in
our implementation. You can use other layer having the same function.

## Example  
For an example, please refer to the models/ directory! The 'example.prototxt'
demonstrates the configuration of a single spatial-IRNN layer.

------------------------------------------------------------------
Please refer to the following papers if you find the source code helpful:  

    @inproceedings{xu2017siamese,  
        Author = {Xu, Xiaqing and Ma, Bingpeng and  Chang, Hong and Chen, Xilin},  
        Title = {Siamese recurrent architecture for visual tracking,  
        Booktitle = {International Conference on Image Processing (ICIP)},  
        Year = {2017}
    }
Contact: hong.chang@vipl.ict.ac.cn  
====================================================================
