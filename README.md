Taking a shot at making machine learning project in Python using PyTorch and other dependencies, to familiarise myself with what AI engineering projects entail.

In this project, program learns how probabilities transition from character to character from a dataset of aggregated names, and uses them to generate new names.

(load dataset -> count pair of neighbouring characters (bigrams) -> normalise by converting into probabilities -> generate new names by sampling characters based on those probabilities)

Day 2:
-> implemented simple neural network in PyTorch to model character level bigrams probabilities.
then, initialised 27x27 weight matrix for 26 letters in alphabet (+. for beginning/end) representing the scores for possible next characters.
-> converted training data into tensors and used matrix operations to process character examples efficiently (that is in batches)
applying exponential transformation and normalisation to convert model scores into actual, useful probability distributions.
-> calculated negative log likelihood loss to measure how well model predicted correct next character.
using gradients and gradient descent to iteratively update the parameters (weights) reducing loss and 'improving' predictions.

Finally, sampled character sequences from trained model to test its output:
ds.
ma.
dichamyieeo.
ddbasa.
esanilimanus.

*yes, very name-ly sounding*


Conclusion:
- I undertook this small project to gain a glimpse behind the iron-curtain of AI and understand the sheer scale of operations fundamental to it. Such, gained practical exposure to the fundamentals like neural network training, tensor operations, probability distributions, loss functions, back-propagation, gradient descent, and character level text generation
- However, it also exposed gaps in my existing knowledge as I frequently found myself referring to PyTorch documentation (most times completely baffled) and even revisiting basic Python concepts. Nevertheless, I deliberately tried to minimise reliance on AI answers, preferring a Socratic approach in order to question, reason and work myself through problems, rather than passively observe solutions.
- With that being said, this is by no means a complete or sophisticated model - and output is far from coherent, but it is undeniably *something*

Going forward:
- Finish my back-end developer course (constantly coding, no passive observation + augmenting with leetcode problems + alongside with maths) -> books (ML for beginners; algorithms in python; data science; ai engineering; TensorFlow) -> Machine learning specialisation coursera -> 3Blue1Brown; StatQuest with Josh Starmer; Andre Keparthi -> Build something that actually yields tangible results
