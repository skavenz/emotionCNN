# emotionCNN
Real-time facial expression tracking system using PyTorch's convolutional neural networks (CNN), OpenCV and PyPlot. Uses a smaller version of the AffectNet dataset which has data of the following expressions: happy, sad, anger, fear, contempt, neutral, disgust and surprise.

To find spatial patterns in the data, I use a CNN - with PyTorch's convolutional, pooling and ReLU (Rectified Linear Unit) layers, these patterns can be identified among every image.
OpenCV is used to feed in live video data as images, preprocessing them immediately and using the trained model to identify the expression. I also utilise its Haar Cascades so that I can crop out the face specifically for preprocessing.
Matplotlib's PyPlot is utilised to visualise the training accuracy and losses, as well as the testing accuracy and losses, ideal to see if the training is going well or needs tuning.

Issues to address: Using PyPlot's subplots, I have identified that the model has overfitting issues as the validation accuracy tends to converge. The training loss constantly goes down consistently as the validation loss is not consistent, meaning the results are less general and are over-tailored towards the training dataset.
