cnntest

how to install digits
	1. git clone https://github.com/NVIDIA/DIGITS.git
	2. digits server "https://localhost" or https://127.0.0.1:5000
	

digits 
	functions: 
		1. create database to train and test
			1.1. create database
				1.1.1 connect to https://127.0.0.1:5000
				1.1.2 move to [datasets]->[new dataset]->[images]->[classification]->[input username-whatever you want]->[set image type and image size]->[move to use image folder]->[set training image->put /path/to/train/data/*.jpg]->[set min sample per class and % for validation]->[set db backend-the way how to treat database]->[set image format]->[set dataset name]->[create]
			1.2. create model -> lenet for mnist db

		2. finetuning
			2.1 train model or get trained model
			2.2 digits->[new model]->[classification]->[select dataset]->[move to previous network and choose one]->[click customize(it will be shown in the right side of the selected network]->[add prototxt]
		3. python layer
			3.1 digits->[new model]->[classification]->[select dataset]->[user side python add *.py file]->[move to previous network and choose one]->[click customize(it will be shown in the right side of the selected network]->[add prototxt]
			3.1.1
					layer {
					  name: "blank_square"
					  type: "Python"
					  bottom: "scale"
					  top: "scale"
					  python_param {
					    module: "digits_python_layers"   #this module name must be same with python file name which defines the player
					    layer: "BlankSquareLayer"
					  }
					  include {
					    phase: TRAIN
					  }
					}


tensorRT
	type: optimizer 
	properties: 
		-use network parameter from caffe and change the parameter to int8 , float16 so that user can reduce the inference time
	
inference : 
	1. download pretrained model
	2. unzip tar.gz(model zip file) in the proper folder
	3. set env_var=/path/to/unzip/file
	4. deploy.prototxt -> forward network architecture, 
		original.prototxt -> original training network architecture
		deploy.prototxt->forward network architecture                         
		snapshot_iter_22620.caffemodel -> weight matrix
		info.json                               
		solver.prototxt -> 
		mean.binaryproto   -> mean image used in training phase                      
		train_val.prototxt -> 

---------------------------------------------------------------------

ps

caffe install
1. Set up the Caffe environment.

   a) Install packages from APT with the following commands:

        $ sudo add-apt-repository universe
        $ sudo add-apt-repository multiverse
        $ sudo apt-get update
        $ sudo apt-get install libboost-all-dev libprotobuf-dev libleveldb-dev libsnappy-dev
        $ sudo apt-get install libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev
        $ sudo apt-get install liblmdb-dev libblas-dev libatlas-base-dev

    b) Download Caffe source package from the following website:

        https://github.com/BVLC/caffe

       And copy the package to $HOME directory on the target board
       with the following command:

        $ mkdir -pv $HOME/Work/caffe
        $ cp caffe-master.zip $HOME/Work/caffe/
        $ cd $HOME/Work/caffe/ && unzip caffe-master.zip

    c) Build Caffe source with the following commands:

        $ cd $HOME/Work/caffe/caffe-master
        $ vi Makefile.config.example

        Uncomment the following line to enable cuDNN acceleration:

            USE_CUDNN := 1

        And modify the following two lines, save, and exit.

            INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
            LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu/hdf5/serial

        $ cp Makefile.config.example Makefile.config
        $ make -j4

        The library libcaffe.so is generated in the build/lib directory.

			
