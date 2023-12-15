# Intrusion detection system using Deep Learning

# Abstract

With the exponential growth in the size of computer networks and application development, the increasing threat of potential damage from cyberattacks becomes more evident. Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS) are emerging as crucial elements in protecting against sophisticated network attacks, continually expanding their capabilities.

This work aims to apply an advanced data modeling method to intrusion detection system data, using deep learning algorithms as a predictive tool for attacks. We utilized the ISCX 2017 dataset provided by the Canadian Institute for Cybersecurity, consisting of seven streams of benign and common attack scenarios, reflecting real-world situations and publicly accessible.

The original dataset comprises 1,580,215 observations over five days, encompassing a diversity of attacks, including 225,745 observations for DDoS attacks, with 85 features. We conducted ten random samplings with cross-validation to create ten data subsets and developed prediction models using deep learning algorithms with a sequential architecture.

The obtained models are compared to other existing techniques by evaluating their accuracy using the confusion matrix. Our model demonstrates an accuracy of 97%.

In conclusion, this document assesses the comprehensive performance of a set of network traffic features using deep learning algorithms to detect attacks in a computer network environment.

# Framework and API
•	Tensorflow-GPU
•	Keras
# Tools
•	Anaconda (Python 3.6)
•	Spyder, Jupyter 
# How to use
Download the ISCX 2017 dataset from the link
https://www.unb.ca/cic/datasets/ids-2017.html
# N.B: If your system is inadequate, I humbly ask you to stop here, as the program won't work efficiently and a lot of time will be wasted.
We have two files: data_processing.py and construction_DeepLearning.py. The first file is used for data pre-processing, graphical representations, checking relevant attributes, and the second file contains the code for building the model.
And now you can start training.
# GOOD LUCK! 
