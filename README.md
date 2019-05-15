# Cognitive-Modeling-Final-Project
Final Project for the course Computational Cognitive Modeling at New York University 

Under Prof. Brenden Lake and Prof. Todd Gureckis

Done by 

Damarapati Mohith
Enaganti Inavamsi Bhaskar
Rajakumar Alfred Ajay Aureate

Abstract


Humans learn to predict patterns in flexible ways.  Pat-terns are generated from a set of rules.  Mathematical se-quences  are  a  kind  of  patterns  with  rules  correspondingto  basic  mathematical  operations  like  addition  and  mul-tiplication.   Once a person understands the mathematicalrules,  she  could  effortlessly  generalize  those  concepts  toother numbers and other combinations of rules.  Althoughthe state-of-the art RNN architectures like LSTMs are con-sidered to perform well on temporal data, they require vastamounts  of  training  examples  and  still  struggle  to  gener-alize  well.   Our  results  show  that  people  does  well  evenwhen  the  sequences  are  slightly  corrupted.   Humans  aresmart enough to disregard the noise and understand the rulethat generated the sequence. Our proposed Bayesian modelcaptures these human-like learning abilities to predict nextnumber in a given sequence.

A sequence is a regularity with its elements repeating ina predictable manner.  Any sequence is built from a certainset of primitives which repeat according to a certain set ofrules. Humans are really good at observing these primitivesand  rules  behind  the  patterns.   Given  a  sequence,  peoplecould naturally predict what comes next by learning rulesbehind  it.   Current  state-of-the-art  Recurrent  Neural  Net-work architectures like Long Short-Term Memory (LSTM)networks require hundreds or thousands of data to predictthe next number in a sequence as they fail to learn richerconcepts which humans do very easily.We would be dealing with strictly increasing mathemat-ical sequences in this project.  Numbers in a mathematicalsequence follow certain rules. In general, a number in a se-quence would depend on the number preceding it.  But, incases like Fibonacci series, a number depends on two num-bers preceding it. We propose a model that captures humanlearning abilities for predicting the next number in the se-quence using Bayesian Concept Learning.Figure 1. Example of caption. It is set in Roman so that mathemat-ics (always set in Roman:BsinA=AsinB) may be includedwithout an ugly clash.Humans are remarkably adaptable to noisy sequences i.e.sequences  with  generational  errors.   Our  model  also  cap-tures this ability of humans to perform well in a noisy en-vironment.  We will be considering noise in our Bayesianmodel while computing likelihood.   This is similar to thehuman approach to capturing the rules of the intended se-quence.People gradually develop their sequence solving skills.At first, they learn to predict sequences which just vary by aconstant addition or multiplication factor. As they develop,they could predict sequences with combinations of multipli-cation and addition factors. And, when their expertise goesto peak, they could even predict sequences which depend ontwo preceding numbers. We are able to capture this graduallearning process of humans in our model with the help of aparameter calledHuman Experience Factor -hef
