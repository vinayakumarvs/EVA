[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%201/Phase-II%20Assignment-01.ipynb)

# Deep Learning for Text

***1.*** Perform One-hot encoding of words and characters

    a. Word level one-hot encoding (toy example): Without Using Keras
    
    b. Character level one-hot encoding (toy example) Without Using Keras
    
    c. Using Keras for word-level one-hot encoding:
    
    d. Word-level one-hot encoding with hashing trick (toy example):
    
***2.*** Using word Embedding

    a. On Keras builtid IMDB data which is Pre Tokenised
    
    b. On Non Tokenised IMDB Data and Tokensize using the GloVe Embeddings
    
    Results are:


<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%201/ResultsGraph/TrainAccuracy.png" width="100%" height="50%">
</centre>

<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%201/ResultsGraph/LossAccuracy.png" width="100%" height="50%">
</centre>
    
    Accuracy Achieved is ~69%
    
    
      By Not Tokenising with GloVe Embeddings
      
      Results are
      
<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%201/ResultsGraph/TrainAccuracyWithoutPreTrain.png" width="100%" height="50%">
</centre>

<img src="https://github.com/vinayakumarvs/EVA/blob/master/Phase%20-%20II/Assignment%201/ResultsGraph/LossAccuracyWithoutPreTrain.png" width="100%" height="50%">
</centre>

     Accuracy Achieved is ~81%

### Conclusion: The Model without loading the pre-trained word embeddings and without freezing the embedding layer has outperformed in comparision to the pre-trained word embeddings.


      
#### Reference: Materials/Deep Learning with Python - https://github.com/vinayakumarvs/EVA/blob/master/Materials/Deep%20Learning%20with%20Python.pdf
