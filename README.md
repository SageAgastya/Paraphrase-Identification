# Assignment:

## Tasks done:
1. The model architecture is ready and working.
2. The postman image is attached and the model is deployed.
3. The image is dockerized and uploaded.
4. After training is done the model will be saved inside a folder named "weights" with .pt extension.
5. Running the functions of TrainAndTest.py will do the training and testing.
6. After training is over, accuracy and F1-score on dev-set will be printed.


## About the code:
1. BertEmbedding.py contains method to get the pretrained bert embeddings.
2. Transformers.py will calculate the self attention over the input pair, followed by cross-attention, called Bahdanau's attention.
3. Paraphrase.py contains Network() class which is our final model.
4. dataclass.py will create Dataset and Dataloader.
5. createconfig.py will create config.json which will contain all the important hyperparameters form TrainAndTest.py.
6. config.json contains hyperparameters from TrainAndTest.py.
7. server.py uses Flask to create API.
8. docker file contains the configuration for docker image.
9. requirements.txt contains all the libraries required for accessing docker-image.
10. weights is a directory which is now empty but this where the trained weights will be saved.
