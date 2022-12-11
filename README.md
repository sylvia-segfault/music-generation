# Piano Music Generation

## Abstract
Natural Language Processing (NLP) models have been used with great success for text generation tasks. Many of these ideas can be extended to other domains. While music may not be considered a spoken language, it bears many similarities, including a notion of time dependent sequential data. In this project, we attempt to use deep learning to compose piano music. Unlike text generation, we attempt to create the music in a form that users can listen to. This would be more akin to training a network to speak the text it creates in addtion to generating the text, which adds an additional challenge. We chose to address this project to explore generation tasks which differ from the text generation project completed in class, but are less compute/time intensive than the image generation tasks we learned about in the final lectures of the class.

Our project will involve finding a suitable set of piano music files to use as training/test data, defining functions for turning the training/test data into tensors, defining a model architecture, generating a music file from tokenized model output, and using human evaluation to determine if the results are good or not. Since music is an art form, it is especially difficult to create an automatic accuracy score for the generated music. As such we will need to try different architectures/hyperparameters and evaluate them manually in order to determine how to adjust the model and parameters to make better music.

## Problem Statement
The specific type of music we are trying to generate is classical style piano music. This will be addressed by the dataset, since if the model trains on classical piano music this is the style of music that it should be generating.
## Related Work
  what papers/ideas inspired you, what datasets did you use, etc
- Talk about being inspired by hw2
## Methodology
  what is your approach/solution/what did you do?
- can mention the choice of a music library (music21 vs. pretty midi)
- mention that converting back to a midi file from the piano roll had noise, it wasn't perfect. A lot of information/notes got lost
  - verified with one input data/a midi file
## Experiements
how are you evaluating your results
- encountered an issue with GPU running out of memory
  - Tried using a smaller dataset which didn't work
    - tiny dataset worked well in the end, from the same composer
  - Making batch size smaller
  - Previously removed test function
- comparison
  - Using the same model, how different temperatures affected the generated music
  - Using the same temperature, how different models affected the generated music
- generated music with many empty notes
  - the model learned the same note had a higher probability, and it ended up holding the same note for a long time
  -  when the frame rate was too high, it had less diversity and there were many duplicate notes
  -  a smaller frame rate trained faster and gave a better result
  -  a tiny dataset + smaller number of samples with a smaller frame did better
## Results
How well did you do
## Examples
images/text/live demo, anything to show off your work (note, demos get some extra credit in the rubric)

#### Markus notes
The dataset and hardware constraints:
NLP models require lots of GPU RAM if large models with large datasets are used, because the recurrent layers keep a notion of history of previously seen notes within the current epoch. Initially, we tried to use ByteDance’s classical piano data set which contains over 10,000 piano songs, but this was too much data for our NVIDIA GeForce GTX 1070 to store in it’s 8GB of GPU RAM. Next, we used one year (2004) of the Maestro Piano dataset. This was a subset of the Maestro dataset and was medium sized, with about 100 piano songs. While we did get it running on the GPU, we had to remove any testing code we had and as such had a difficult time evaluating the model. Also, we had to use a very tiny batch size which we believe negatively effected our model’s ability to learn patterns in the music, leading to very bad generation results. Finally, in order to refocus our project and attempt to get good results, we downloaded a small dataset of German composer Robert Schumann of about 10 songs. This finally got us some results that sounded good, and also weren’t just instances of the model memorizing the given songs.

Code:
In order to complete our project within a reasonable amount of time, we tried to reuse much of the given HW2 code so we wouldn’t spend time debugging training infrastructure related bugs and could instead focus on processing our dataset,  model architecture, and music file regeneration algorithms. We did need to completely rewrite the HW2 data preprocessing code, but did so in a way that we could still make use of the HW2 Dataset and Vocabulary classes. Our preprocessing code makes use of Python multithreading in order to tokenize midi files in parallel, which had challenges such as locking and releasing shared data structures during critical sections. Parallelizing this code gave us a huge speedup in data preprocessing time and was well worth the initial investment.

We also optimized beam sampling to work in parallel. Unlike HW2, beam sampling actually gave us some of our best results, and was a bottleneck during the generation process. We greatly reduced the time lost during beam sampling by parallelizing this code as well.

We used a similar final model structure as HW2, with an encoder/decoder structure but instead of a GRU we used two stacked LSTM modules between the encoder decoder models. This stack gave us a fairly complex model but allowed us to start with the good hyper parameters given in HW2 and PyTorch handle the intricacies of the model itself.

We also needed to update the generate_language function to take our tokens and return a midi file instead of just text, which was more complex than the generation process we saw in HW2. We again using pretty_midi to create the files, and found open source code online which provided a function for converting our chords back into a format pretty_midi could work with.


Our project was challenging in many ways. The most important challenge was figuring out how to represent the music in a way that our model could use. We decided to use the pretty_midi python library to help us break down midi music files into NumPy arrays which we could then process using Python. Our vocabulary consisted of piano chords, which for us meant any combination of one or more piano keys held down during a given time. Whist processing our dataset, which one of these unique chords we discovered was added to the vocabulary, in addition to the empty note, which is a chord with none of the keys pressed down. This meant that our vocabulary could be extremely large, since there are 88 keys on a piano. Luckily, humans have ten fingers so we didn’t expect each chord to consist of more than ten notes, and human hands can’t stretch across the entire keyboard, so chords consisted of keys humans could reasonably reach with 2 hands. Given that many combinations of notes don’t sound great, this actually made our vocabulary smaller than you might expect (Note that there might be more chords if songs in our dataset were duets or involved more than one piano, we’re not sure if this was the case or not).

Midi files have a frame rate, which is the granularity of the notes which can be played in one second. A high frame rate may be required in practice to represent songs whose tempo may not be aligned with natural seconds. The music will sound more natural the higher the frame rate used. We initially wanted to try using a higher frame rate with our data preprocessing and training but we found that this yielded negative results. We used sequences of length 100 (aka 100 chords, one chord per frame), and when a high frame rate was used, we saw a lot of duplicated notes. For example, if we used a frame rate of thirty and a chord was held for one second, 30 of 100 tokens in our sequence would be the same. When training, this cause our model to just hold to just play the same note over and over as generally holding down the a similar note caused relatively low loss. We finally solved this problem by preprocessing our data using a frame rate of two. This caused the sequence of length 100 to consist of many different tokens rather than just repeating one, which led to way better models that did not repeat one note as much. The downside of this was our music was always a bit slow. We were able to solve this problem by taking the music generated by our model and upscaling the frames per second from two to eight when converting back into a midi file. This sped up the music, and we could simply generate more notes in place of the time lost by using this speedup. We decided to user eight since it was a power of two that seems like a fair number of chords that an intermediate piano player can make in a single second. Naturally, the songs we generated weren’t extremely fast songs, but we did get nice slower melodies that are pleasant to listen to.



Future:

Maybe could achieve a higher framerate with an encoder decoder structure, which removes noise from music with a high frame rate.
