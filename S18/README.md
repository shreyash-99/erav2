# Session 18 Assignment 

### Problem Statement 
1. Pick the "en-it" dataset from opus_books
2. Train your transformer (encoder-decoder) (do anything you want, use PyTorch, OCP, AMP, etc), but get your loss under 1.8 within 18 Epochs. So of course you have to accelerate your transformer a lot!
3. Enjoy! 

### Progress Till Now 
- In the last assignment, we trained the basic encoder-decoder transformer network. The training speed was very and it took around 22 min per epoch in T4. 
- Reached an accuracy of 2.717 after 14 epochs 

## Implementing Dynamic Batching 
- This proved to be the best way to reduce the time taken for training at every epoch. 
- While training of batch it doesnt matter what are the first 2 dimentions of matrix(which are the batch size or number of sentences and seq_len or number of tokens) and therefore these 2 dimentions can be different in different rounds of training. 
- Earlier we were keeping the value of seq_len as more than the length of the largest sentence of the training dataset.
- This is not necessary as very small number of the sentences are this long, and for the small ones the length till the seqlen is filled with padded token, which when passed through the attention layer consumes lot of time.
#### How to implement this
1. First i sorted the raw dataset, so that a batch of data will sort of same length of data. So that there will be minimum number of pad tokens used, so that minimum computation is begin wasted.
2. When every sentence of passed through the BillingualDataset class, the encoder_input, decoder_input, label only comprised of the tokens that are neccesary and doesn't include the padded tokens.
3. When the data is being loaded and converted to batches, the batch of data is passed through collate function. the encoder_input and decoder_input is then appended with pad tokens based the length of the longest sentence in the group.
4. This batch is then passed for the training

### Results
1. Training time decreased from 22 min to around 6 min for an epoch
2. Reach training accuracy of 2.21 in 18 epochs
3. [Python Notebook Implementing Dynamic Batching(with training logs)](/S18/S18-Dynamic_Batching.ipynb)


## Automatic Mixed Precision
Used for decreasing the time taken for training. Doing all the multiplication in float32 is computationally expensive therefore by using AMP, the  multiplication which attention calculations is carried out is done in float16 while the important calculations like backpropagation is still executed in float32

## One Cycle Policy
This is implemented to reach a lower accuracy faster, it doesnt help in increasing the accuracy of a model but only reaching that number faster.
- used starting MAX_LR = 10**-3 (ten times the lr mentioned in the official paper)
- starting and ending division factor is 100
- max lr at 30% of the epochs

### How is it implemented

### Results
1. Decreased the training time from 6 minutes to about 4 minutes 30 seconds.
2. Reached a loss of 1.81 in the 18th epoch.
3. [Python notebook - Dynamic Batching + One Cycle Policy + Automatic Mixed Precision](/S18/S18_DB_AMP_OCP.ipynb)

### Logs
Processing Epoch 00: [04:30<00:00,  4.48it/s, recent loss=6.145, avg loss=6.899, lr=[0.00020786939313984168] <br>
Processing Epoch 01: [04:29<00:00,  4.50it/s, recent loss=5.415, avg loss=5.713, lr=[0.0004059020448548813]]<br>
Processing Epoch 02: [04:25<00:00,  4.57it/s, recent loss=5.420, avg loss=5.291, lr=[0.0006039346965699209]]<br>
Processing Epoch 03: [04:26<00:00,  4.55it/s, recent loss=4.896, avg loss=4.974, lr=[0.0008019673482849604]]<br>
Processing Epoch 04: [04:26<00:00,  4.56it/s, recent loss=4.649, avg loss=4.687, lr=[0.000999836741424802]]<br>
Processing Epoch 05: [04:27<00:00,  4.54it/s, recent loss=4.378, avg loss=4.381, lr=[0.0009231480246052381]]<br>
Processing Epoch 06: [04:27<00:00,  4.54it/s, recent loss=3.993, avg loss=4.010, lr=[0.0008462960492104763]]<br>
Processing Epoch 07: [04:25<00:00,  4.57it/s, recent loss=3.639, avg loss=3.652, lr=[0.0007694440738157144]]<br>
Processing Epoch 08: [04:26<00:00,  4.55it/s, recent loss=3.325, avg loss=3.309, lr=[0.0006925286892003298]]<br>
Processing Epoch 09: [04:26<00:00,  4.55it/s, recent loss=3.249, avg loss=3.003, lr=[0.0006156767138055679]]<br>
Processing Epoch 10: [04:27<00:00,  4.54it/s, recent loss=2.609, avg loss=2.746, lr=[0.0005387613291901834]]<br>
Processing Epoch 11: [04:24<00:00,  4.59it/s, recent loss=2.449, avg loss=2.522, lr=[0.0004618459445747987]]<br>
Processing Epoch 12: [04:25<00:00,  4.57it/s, recent loss=2.180, avg loss=2.338, lr=[0.00038493055995941403]]<br>
Processing Epoch 13: [04:24<00:00,  4.59it/s, recent loss=2.207, avg loss=2.181, lr=[0.00030807858456465215]]<br>
Processing Epoch 14: [04:26<00:00,  4.55it/s, recent loss=1.929, avg loss=2.052, lr=[0.00023122660916989027]]<br>
Processing Epoch 15: [04:25<00:00,  4.56it/s, recent loss=1.857, avg loss=1.952, lr=[0.0001543112245545057]]<br>
Processing Epoch 16: [04:25<00:00,  4.57it/s, recent loss=1.955, avg loss=1.870, lr=[7.745924915974394e-05]]<br>
Processing Epoch 17: [04:24<00:00,  4.58it/s, recent loss=1.846, avg loss=1.813, lr=[5.438645443592641e-07]]<br>