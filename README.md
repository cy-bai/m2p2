# Code and dataset for M2P2

----
## Code
### Dependencies
python 3: pytorch, numpy, scipy, pandas. GPU-version only.

### How to run
Download *qps\_dataset.tar.gz*, extract it, and put the extracted *qps\_dataset* folder under the ppami folder.

Create a folder to save your own trained models:
>block quote

    mkdir new_trained_models
To test the performance of our pre-trained model, run
>block quote

    python main.py --test_mode --fd=FOLD
The MSE loss in the test set of fold FOLD will be printed.

To train the model we proposed, run
>block quote

    python main.py --het_module --fd=FOLD --verbose
This will train the model for fold FOLD and output the concat weights as well as training and validation loss every 2 epochs (if enabling *verbose*), the trained model and concat weights  will be saved in *new\_trained\_models/fold[FOLD]/*.

### Code details
* main.py. Executes the training / testing of the whole work.

* dataset.py. Loads raw features from all modalities to pytorch.

* model.py. Our model for learning latent embeddings, making predictions and reference models.

* train.py. Training and Evaluation functions for all prediction models and reference models.

* utils.py. Utility function, defined constants and hyperparameters.

* folds_split/. Stores the segments used for training, validation and test in each fold.

* models/. Save pre-trained models and concat weights for each fold.

## QPS dataset
*qps\_index.csv* stores the index of the whole dataset, including the debate episode ID, clip ID, segment ID, change of votes, end vote, clip length (dur_sec), etc.

Extract *qp\_dataset.tar.gz* to get the dataset.  Inside, each folder represents a segment corresponding to *qps\_index.csv* by *seg\_id* column. 

In each segment folder, *covarep\_norm.npy* is the extracted COVAREP audio features, *tencent\_emb.npy* is the extracted word embeddings, and *vgg_1fc* stores the extracted features of each frame extracted from the pre-trained CNN without the last fc layer. We use these features as input to get the input embeddings.

**Note that we have only released the raw features. We can't release the raw videos right now due to iQIYI Privacy Policy, we are working on it, and we will release once getting their approval.**
