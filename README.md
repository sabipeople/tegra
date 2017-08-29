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
			
