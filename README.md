# Classic Piano Music Generation
![0*DkBU-0utTPNALjmT](https://user-images.githubusercontent.com/72118815/206929235-00061ca1-c9bf-4cf9-9b59-6a2a567b0de4.jpg)


## Abstract
Natural Language Processing (NLP) models have been used with great success for text generation tasks. Many of these ideas can be extended to other domains. While music may not be considered a spoken language, it bears many similarities, including a notion of time dependent sequential data. In this project, we attempt to use deep learning to compose classic piano music. Unlike text generation, we attempt to create music in a form that users can listen to. Text generation trains a neural network that can imitate language speaking or writing a story. In contrast, music generation adds an addtional challenge of translating abstract art to something that can be understood by a deep learning model. We chose to address this project and explore generation tasks which differ from the text generation project completed in class. Music generation is as computationally intensive as the image generation tasks.

## Problem Statement
The specific type of music we are trying to generate is classical style piano music. This will be addressed by the dataset, since if the model trains on classical piano music this is the style of music that it will learn to generate. Our project will involve finding a suitable set of piano music files as the training/test data, defining functions for turning the training/test data into tensors, defining a model architecture, generating a music file from tokenized model output, and using human evaluation to determine if the results are good or not. Since music is an art form, it is especially difficult to create an automatic accuracy score for the generated music. As such we will need to try different architectures/hyperparameters and evaluate them manually in order to determine how to adjust the model and parameters to make better music.

## Related Work
We were inspired by text generation in terms of tokenizing the training data as well as using different sampling strategies to generate the results.

## Demo
https://user-images.githubusercontent.com/72118815/207254704-e8f4c18d-fc41-4bc6-91c7-3753aceba31d.mp4

## Datasets
[Big Dataset](https://github.com/bytedance/GiantMIDI-Piano)  
[Medium Dataset](https://www.kaggle.com/datasets/kritanjalijain/maestropianomidi)  
[Small Dataset](http://www.piano-midi.de/schum.htm)  

## Methodology

### Processing and Tokenizing the Datasets
In order to complete our project within a reasonable amount of time, we tried to reuse the text generation code so we wouldn’t spend time debugging training infrastructure related bugs and could instead focus on processing our dataset, model architecture, and music file regeneration algorithms. We had to completely rewrite the data preprocessing code, but did so in a way that we could still make use of the `Vocabulary` class. Our preprocessing code utilized Python multithreading in order to tokenize midi files in parallel, which had challenges such as locking and releasing shared data structures during critical sections. Parallelizing this code gave us a huge speedup in data preprocessing time and was well worth the initial investment.

The most important challenge was figuring out how to represent the music in a way that our model could use. We used midi files as our dataset beocause they occupy very little memory and are a good representation of classic piano music. We broke down midi music files into NumPy arrays, a 2D matrix with 128 rows for all the possible piano notes and `n` columns where `n` is the number of time frames in a song, using [`pretty_midi`](https://gist.github.com/gudgud96/e925e47fe781e0573e4224cb3d085cbf) python library. We created a `Vocabulary` class consisted of piano chords, which represents a combination of one or more piano keys played during a given time. The piano chords were stored in a tuple. While processing our dataset, we added unique chords to a dictionary mapping the timeframes when those chords were played in each song of the datasets. The timeframes were then used to tokenize every song in the datasets. It was possible that at some timeframes, no chords were played in which case we used an empty tuple for its representation. As a result, our vocabulary could be extremely large, since there are 88 keys on a piano. Luckily, humans have ten fingers so we didn’t expect each chord to consist of more than ten notes, and human hands can’t stretch across the entire keyboard, so chords contained keys humans could reasonably reach with 2 hands. Given that many combinations of notes don’t sound great, this made our vocabulary smaller than expected. There might be more chords if songs in our datasets were duets or involved more than one piano, however, we were not sure whether that was the case or not.

### Model Architecture
After experimenting with a few architectures, we ended up wrapping two stacked LSTM modules between an encoder and a decoder. This structure gave us a fairly complex model that allowed us to tune hyperparameters from a good start, and take the advantage of PyTorch handling the intricacies of the model itself. The hyperparameters were:
- `temperature = 0.9`,  for scaling the probability distribution of unique chords
- `input fs = 2`,  frame per second for input training data
- `output fs = 8`,  frame per second for generaing classic piano music
- `sequence_length = 100`,  the length of each partially tokenized song
- `batch_size = 256`,  the size of the training samples in each batch
- `feature_size = 512`,  number of features fed to the LSTM model
- `epochs = 50`,  number of epochs to train
- `learning_rate = 0.002`
- `weight_decay = 0.0005`

### Music Generation Algorithms
We used max sampling strategy, sample sampling strategy and beam sampling strategy. The beam sampling was optimized so that beams were sampled in parallel. It produced some of our best results, but was a bottleneck during the generation process. We greatly reduced the time lost during beam sampling by parallelizing it. We wrote a generation function to produce classic piano music using the trained deep learning model, which was tokenized. Then we converted it into a piano roll. We used this reference [`piano_roll_to_pretty_midi`](https://gist.github.com/gudgud96/e925e47fe781e0573e4224cb3d085cbf) that uses `pretty_midi` library to convert each piano roll back to a midi file, since the `pretty_midi` library doesn't have a function that performs the conversion. However, this conversion wasn't perfect and introduced a lot of noise. We tested using a midi file from our dataset and manually verified that many chords and sounds were lost after:
- converting it to a piano roll using the `get_piano_roll` function from the pretty_midi library
- converting the piano roll back to a midi file

## Experiments
### The Dataset and Hardware Constraints
NLP models require lots of GPU RAM if large models with large datasets are used, because the recurrent layers keep a notion of history of previously seen notes within the current epoch. Initially, we tried to use ByteDance’s classical piano data set which contains over 10,000 piano songs, but this was too much data for our NVIDIA GeForce GTX 1070 to store in it’s 8GB of GPU RAM. Next, we used one year (2004) of the Maestro Piano dataset. This was a subset of the Maestro dataset and was medium sized, with about 100 piano songs. While we did get it running on the GPU, we had to remove any testing code we had and as such had a difficult time evaluating the model. Also, we had to use a very tiny batch size which we believe negatively effected our model’s ability to learn patterns in the music, leading to very bad generation results. Finally, in order to refocus our project and attempt to get good results, we downloaded a small dataset of 8 songs by the German composer Robert Schumann. This finally generated some classic piano music that sounded better with more melody, rhythm and variations of chords, which weren’t just instances of the model memorizing the given songs.

### Model Architecture
For the same temperature, a LSTM model performed better than a GRU model. We used **2 recurrent nueral network layers** for LSTM since stacking RNNs on top of each other allowed the model to learn different chords of the training songs by extracting more features. We also experiemented a lot with how temperature changed the generated music for the same model (using LSTM). We started with **temperature=1.5** which sounded like a child smashing piano keys violently. However, temperature=0.7 mostly produced music that only had **1** or **2** notes. With this observation, We fine-tuned the temperature by graduating reducing the temperature from **1.5, to 1.2, then 0.99, and eventually 0.9**. **temperature=0.9** gave us the best sounded music.

### Low Frame Rate Trained the Model to Play More Diverse Chords
Midi files have a frame rate, which is the granularity of the notes which can be played in one second. A high frame rate may be required in practice to represent songs whose tempo may not be aligned with natural seconds. The music will sound more natural the higher the frame rate used. We initially wanted to try using a higher frame rate with our data preprocessing and training but we found that this yielded negative results. We used **sequences of length 100 (aka 100 chords, one chord per frame)**, and when a high frame rate was used, we saw a lot of duplicated notes. For example, if we used a frame rate of thirty and a chord was held for one second, **30 of 100 tokens** in our sequence would be the same. During the process of training, this caused our model to repeat the same note over and over again as generally holding down a similar note caused relatively low loss. We finally solved this problem by preprocessing our data using **a frame rate of 2**. This caused the sequence of length 100 to consist of many different tokens rather than just repeating one, which led to way better models that played more diverse chords. The downside of this was that generated music was a bit slow. We were able to solve this problem by taking the music generated by our model and upscaling the frames per second from **2 to 8** when converting back into a midi file. This sped up the music, and we could simply generate more notes in place of the time lost by using this speedup. We decided to use **8** since it was a power of **2** that seemed to give a fair number of chords that an intermediate piano player could make in a single second. Naturally, the songs we generated weren’t extremely fast songs, but we did get nice slower melodies that are pleasant to listen to.

### Experiment Results
This Google Drive contains all the trials for our experiments. The `try6_best` subfolder in the `final exp` folder produced the best result.

[Experiment Results](https://drive.google.com/drive/folders/1--1CWX1cx5vBNKhq1ccjppwZvsG8t1Ny?usp=sharing)

## Results
[Generated Music in MIDI Format](https://drive.google.com/drive/folders/1BSbhdEFyr-uAaAIe3_LxYAs49AToQ0fm?usp=sharing)

### MP4 Version
We used an online free converter to convert music files from MIDI format to MP4. However, the music sounded different after converting to the MP4 version. We would recommend listenting to the music files in MIDI format instead.   
   
[max sample](https://user-images.githubusercontent.com/72118815/206929298-c10f5e97-d599-4d87-8f6c-47e2c4b0eb7c.mp4)    
[sampling sample](https://user-images.githubusercontent.com/72118815/206929300-4039df5e-7ef1-46c2-9a9c-ed42494ff8c4.mp4)    
[beam sample 0](https://user-images.githubusercontent.com/72118815/206929303-21df1f1f-1fce-455c-8b29-ef90229985da.mp4)         
[beam sample 1](https://user-images.githubusercontent.com/72118815/206929305-4debf615-649c-4d7b-b5f1-cf1dd2baf56e.mp4)    
[beam sample 2](https://user-images.githubusercontent.com/72118815/206929306-e5b8d8fd-2789-4618-9009-678e899026b7.mp4)    
[beam sample 3](https://user-images.githubusercontent.com/72118815/206929307-e021f8a6-39ed-4141-ba92-b5dd887991b3.mp4)    
[beam sample 4](https://user-images.githubusercontent.com/72118815/206929311-d1687230-b22d-4656-af41-65bbb0a2aa82.mp4)    

### Visualization
#### Piano Roll
Here are some of our favorite examples for visualizing the generated piano roll by our deep learning model. This shows the notes in a MIDI region as colored bars in a time grid. Horizontal lines show the time position, while vertical lines indicate pitch. A keyboard along the left edge provides a reference for the pitches of notes.

<img width="1434" alt="Screen Shot 2022-12-11 at 1 27 41 PM" src="https://user-images.githubusercontent.com/72118815/206930013-8664130f-126b-48c4-8cea-0244b16b156d.png">
<img width="1434" alt="Screen Shot 2022-12-11 at 1 28 29 PM" src="https://user-images.githubusercontent.com/72118815/206930016-21442489-3400-479b-b7ca-83c7c2c94cff.png">

#### Plots
Please click on each plot to see titles and axis!
![accuracy-2](https://user-images.githubusercontent.com/72118815/206930210-e2ff335b-ec77-4760-916a-d5b7b0f92423.png)
![testloss-2](https://user-images.githubusercontent.com/72118815/206930221-6d334cfa-03d3-4b49-b5c9-c2bdc819fcd5.png)
![testperp-2](https://user-images.githubusercontent.com/72118815/206930223-53fd8a67-98ba-4cf4-92d7-82a249ee6811.png)
![trainloss-2](https://user-images.githubusercontent.com/72118815/206930226-98a976a3-ed80-4402-bdad-f5981728a857.png)
![trainperp-2](https://user-images.githubusercontent.com/72118815/206930227-8c4a06fb-1482-41a5-9414-f9e8f241352f.png)

## Future Work
- Our approach in terms of preprocessing the input data was to combine each tokenized song together and then use them as data and label for the training process. Some hypotheses are that if we have enough GPU resources, we could train on a larger dataset and take partial notes from each song. Then we can shuffle those partially tokenized songs which might give us better results.
- We could also train another deep learning model that learns how to add and remove noise from music with an encoder and a decoder. This allows us to use a higher framerate.
