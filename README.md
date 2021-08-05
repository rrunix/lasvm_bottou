Project description
===================

This project is part of a research project where lasvm was used as a baseline. Thus, this code was not mean to be used in production or any other setting. Motivations:

* The original code despite providing an online training interface loads all the data at the begining.
* Therefore, while being online, it can not handle retraining used new observations.
* This project provides a hacky way to simulate retraining. Similarly to the original implementation, all the data is provided at the begining.
* However, it allow to specify which range of data belongs to each retraining, and it stores the models for each retraining.
* In short, it can simulate retrainings but all the data is provided at the begining, and consequently, it has no practical use other than test the model.
* In a traditional setting where no retrainings are needed, the model can be trained with all the data and make predictions as usual.


Usage
=====

To compile the project open a terminal inside the project folder an run:

```bash
make 
```

To see an usage example see la_svm_example.py

LaSVM
=====

![LaSVM comparision graph](http://leon.bottou.org/_media/papers/lasvm-epsiloncache.png?w=300&tok=a93ee2)

LASVM is an approximate SVM solver that uses online approximation. It reaches accuracies similar to that of a real SVM after performing a single sequential pass through the training examples. Further benefits can be achieved using selective sampling techniques to choose which example should be considered next.

As show in the graph, LASVM requires considerably less memory than a regular SVM solver. This becomes a considerable speed advantage for large training sets. In fact LASVM has been used to train a 10 class SVM classifier with [8 million examples](http://leon.bottou.org/papers/loosli-canu-bottou-2006) on a single processor.

See the [LaSVM paper](http://leon.bottou.org/papers/bordes-ertekin-weston-bottou-2005) for the details.

## Source

http://leon.bottou.org/projects/lasvm


## Credits
All credits to the original implementation. 