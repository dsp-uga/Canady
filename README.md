# Project 3 - Neuron Finding 
## Team-Canady
### Members:
* Ailing Wang
* Hiten Nirmal
* Vibodh Fenani
* Vyom Shrivastava

## Technology Used:
* Python

* Apache Spark

* Keras

* Thunder NMF Extraction

## Problem Statement

Calcium imaging is a technique for observing neuron activity as a series of images showing indicator fluorescence over time. Manually segmenting neurons is time-consuming, leading to research on automated calcium imaging segmentation. We want to design a model that will handle a high-dimensional, large-scale image segmentation problem. Our goal is to locate the neurons and segment them out from the surrounding image, given large sets of tiff images, as accurately as possible.


## Dataset
The datasets we used to train and test is provided by Dr. Shannon Quinn for the course CSCI 8360: Data Science Practicum.

There are total 9(test) Dataset which are being evaluated.

Training and Testing Dataset can also be found on below website:

[Neuro Finder website](http://neurofinder.codeneuro.org/)


## Execution Steps

The project requires the following technologies to be installed.
* Instructions to download and install Python can be found [here](https://www.python.org/).
* Instructions to download and install Apache Spark can be found [here](https://spark.apache.org/docs/latest/).
* Instructions to download and install Keras can be found [here](https://keras.io/).
* Instructions to download and install Thunder and Thunder extraction can be found [here](https://github.com/thunder-project/thunder)

## NMF Flow

* Use thunder library and import that in your code.
* Load the testing dataset.
* Create the algorithm with various parameters.
* Fit the model in our algorithm.
* Transform and merge the overlapping coordinates.
* Save the output in desired format.
* Goto [Wiki](https://github.com/dsp-uga/Canady/wiki/NMF_Thunder_Model) for more detailes.

## Approaches we tried

We used U-net and thunder extraction module for neuron segmentation.

## Accuracy
----to be filled

## References

* http://neurofinder.codeneuro.org/
* https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
* https://github.com/thunder-project/thunder-extraction
* https://github.com/thunder-project
* https://arxiv.org/pdf/1707.06314.pdf- Research paper for U-net

## License

This project is licensed under the MIT License - see the [License](https://github.com/dsp-uga/Canady/blob/master/LICENSE) file for details


