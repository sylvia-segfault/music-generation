# TITLE GOES HERE

## Abstract 
in a paragraph or two, summarize the project

## Problem Statement
what are you trying to solve/do
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
