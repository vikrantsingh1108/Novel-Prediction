# Novel-Prediction
Predicting a particular text belonging to a certain Novel 
I have used 2 ConvNets with one large and one small. They are both 9 layers deep with 6 convolution layers and 3 fully-connected layers.

conv_layers = [
                    [256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
                    ],

	Each convolution layer has 256 small features with a kernel size 7 and pool size 3

                 fully_layers = [1024, 1024],
	Two fully connected layer is used and one depending on the problem, in this case we have 12 classes hence it will be 12. Dropout has been used in fully-connected layers for regularization.

Results:
	The character level CNN was much better than the other models. 
	Training Accuracy : 85%
	Testing Accuracy : 71%

Reference: xiang Z. et.al.,(2015) Advances in Neural Information Processing Systems 28 (NIPS 2015). DOI arXiv:1509.01626
