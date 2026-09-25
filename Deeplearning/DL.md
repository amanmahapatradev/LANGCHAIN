Deep Learning ---> Machine Learning ---> AI ---> ML Operations ---> Data Science ---> Big Data ---> Data Analytics
Works with unstructured Data (Text, Image, Audio, Video) 
Hendle complex operations(image_classification, video_generation, etc) 
Feature Extraction 
Achieve best performance in NLP tasks(image-captioning, speech-recognition, etc)  

PERCEPTRON -> MUltilayer perceptron -> Feedforward neural network -> Recurrent neural network -> Convolutional neural network -> Generative adversarial network -> Transformers -> Deep learning model -> Deep learning model 

Perceptron -> AND, OR Gate -> NOT Gate -> XOR Gate -> Multilayer Perceptron 

Neural Networks -> Hidden Layers -> Nodes -> Activation function -> Weights -> Bias

Types Neural Networks -> Feedforward Neural Network -> Recurrent Neural Network -> Convolutional Neural Network -> Generative Adversarial Network -> Transformers -> Deep learning model 

Application of Deeplearning -> 
Computer Vision -> image-classification, video-generation, etc 
NLP -> image-captioning, speech-recognition, etc 
Generative AI -> text-generation, image-generation, etc 
Speech Recognition -> image-captioning, speech-recognition, etc 

Cost function -> Mean Square Error(MSE) -> Cross Entropy -> Negative log-likelihood

Feed Forward Neural Network -> Input Layer -> Hidden Layer -> Output Layer 

Application of RNN ->  language-modelling, machine-translation, speech-recognition, etc
Application of LSTM ->  language-modelling, machine-translation, speech-recognition, etc
Application of GRU ->  language-modelling, machine-translation, speech-recognition, etc
Application of CNN ->  image-classification, video-generation, etc 
Application of GAN ->  image-classification, video-generation, etc 
Application of Transformers ->  image-classification, video-generation, etc 
Application of Deep learning model ->  image-classification, video-generation, etc 

RNN looks like -> Input -> Hidden -> Output  <-- Hidden --> Output  

RNN ->LSTM -> GRU -> LSTM -> GRU   ->  Transformers -> Deep learning model 

Cost function -> Mean Square Error(MSE) -> Cross Entropy -> Negative log-likelihood 

Optimizers -> Batch Gradient Descent -> Stochastic Gradient Descent (SGD) -> Mini-batch Gradient Descent  

Vanishing Gradient problem -> Sigmoid, Tanh -> ReLU -> Leaky ReLU -> ELU -> SELU 
Explaining Gradient Problem -> 
        Input
            | 
        Hidden Layer 1
            | 
        Hidden Layer 2
            | 
        Hidden Layer 3
            | 
        Output Layer    

        Output Layer -> Hidden Layer 3 -> Hidden Layer 2 -> Hidden Layer 1 -> Input Layer   

        Output Layer -> Hidden Layer 3 -> Hidden Layer 2 -> Hidden Layer 1 -> Input Layer   
                | 
                |
                | 
        Output Layer 
                    |  
                    | 
        Hidden Layer 3 
                    |  
                    | 
        Hidden Layer 2 
                    |  
                    | 
        Hidden Layer 1 
                    | 
                    | 
        Input Layer 


Solution to Gradient Problem -> 
    1. Batch Gradient Descent -> 
        - Computes gradients over the entire dataset 
        - Stable but slow 
    2. Stochastic Gradient Descent (SGD) -> 
        - Computes gradients per individual sample 
        - Fast, noisy 
    3. Mini-batch Gradient Descent -> 
        - Computes gradients per small batch (32, 64, 128, etc.) 
        - Standard default for most tasks 