# cs-224n-assignment-3-dependency-parsing-solved
**TO GET THIS SOLUTION VISIT:** [CS 224n Assignment #3: Dependency Parsing Solved](https://www.ankitcodinghub.com/product/cs-224n-assignment-3-dependency-parsing-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;49398&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;6&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (6 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS 224n Assignment #3: Dependency Parsing Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (6 votes)    </div>
    </div>
In this assignment, you will build a neural dependency parser using PyTorch. In Part 1, you will learn about two general neural network techniques (Adam Optimization and Dropout) that you will use to build the dependency parser in Part 2. In Part 2, you will implement and train the dependency parser, before analyzing a few erroneous dependency parses.

<h1>1.&nbsp;&nbsp;&nbsp;&nbsp; Machine Learning &amp; Neural Networks (8 points)</h1>
<ul>
<li>(4 points) Adam Optimizer</li>
</ul>
Recall the standard Stochastic Gradient Descent update rule:

<em>θ </em>← <em>θ </em>− <em>α</em>∇<em>θJ</em><sub>minibatch</sub>(<em>θ</em>)

where <em>θ </em>is a vector containing all of the model parameters, <em>J </em>is the loss function, ∇<em>θJ</em><sub>minibatch</sub>(<em>θ</em>) is the gradient of the loss function with respect to the parameters on a minibatch of data, and <em>α </em>is the learning rate. Adam Optimization<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> uses a more sophisticated update rule with two additional steps.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

<ol>
<li>(2 points) First, Adam uses a trick called <em>momentum </em>by keeping track of <strong>m</strong>, a rolling average</li>
</ol>
of the gradients:

<strong>m </strong>← <em>β</em>1<strong>m </strong>+ (1 − <em>β</em>1)∇<em>θJ</em>minibatch(<em>θ</em>) <em>θ </em>← <em>θ </em>− <em>α</em><strong>m</strong>

where <em>β</em><sub>1 </sub>is a hyperparameter between 0 and 1 (often set to 0.9). Briefly explain (you don’t need to prove mathematically, just give an intuition) how using <strong>m </strong>stops the updates from varying as much and why this low variance may be helpful to learning, overall.

<ol>
<li>(2 points) Adam also uses <em>adaptive learning rates </em>by keeping track of <strong>v</strong>, a rolling average of the magnitudes of the gradients:</li>
</ol>
<strong>m </strong>← <em>β</em>1<strong>m </strong>+ (1 − <em>β</em>1)∇<em>θJ</em>minibatch(<em>θ</em>)

<strong>v </strong>← <em>β</em>2<strong>v </strong>+ (1 − <em>β</em>2)(∇<em>θJ</em>minibatchminibatch(<em>θ</em>))

<strong>v</strong>

where &nbsp;and <em>/ </em>denote elementwise multiplication and division (so <strong>zz </strong>is elementwise squaring) and <em>β</em><sub>2 </sub>is a hyperparameter between 0 and 1 (often set to 0.99). Since Adam divides the update

√

by&nbsp; <strong>v</strong>, which of the model parameters will get larger updates? Why might this help with learning?

<ul>
<li>(4 points) Dropout<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> is a regularization technique. During training, dropout randomly sets units in the hidden layer <strong>h </strong>to zero with probability <em>p</em><sub>drop </sub>(dropping different units each minibatch), and then multiplies <strong>h </strong>by a constant <em>γ</em>. We can write this as</li>
</ul>
<strong>h</strong><sub>drop </sub>= <em>γ</em><strong>d </strong>◦ <strong>h</strong>

where <strong>d </strong>∈ {0<em>,</em>1}<em><sup>D</sup></em><em><sup>h </sup></em>(<em>D<sub>h </sub></em>is the size of <strong>h</strong>) is a mask vector where each entry is 0 with probability <em>p</em><sub>drop </sub>and 1 with probability (1 − <em>p</em><sub>drop</sub>). <em>γ </em>is chosen such that the expected value of <strong>h</strong><sub>drop </sub>is <strong>h</strong>:

E<em>p</em><sub>drop</sub>[<strong>h</strong><em>drop</em>]<em>i </em>= <em>h</em><em>i</em>

for all <em>i </em>∈ {1<em>,…,D<sub>h</sub></em>}.

1

<ol>
<li>(2 points) What must <em>γ </em>equal in terms of <em>p</em><sub>drop</sub>? Briefly justify your answer. ii. (2 points) Why should we apply dropout during training but not during evaluation?</li>
</ol>
<h1>2.&nbsp;&nbsp;&nbsp;&nbsp; Neural Transition-Based Dependency Parsing (42 points)</h1>
In this section, you’ll be implementing a neural-network based dependency parser, with the goal of maximizing performance on the UAS (Unlabeled Attachemnt Score) metric.

Before you begin please install PyTorch 1.0.0 from <a href="https://pytorch.org/get-started/locally/">https://pytorch.org/get-started/locally/ </a>with the CUDA option set to None. Additionally run pip install tqdm to install the tqdm package – which produces progress bar visualizations throughout your training process.

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between <em>head </em>words, and words which modify those heads. Your implementation will be a <em>transition-based </em>parser, which incrementally builds up a parse one step at a time. At every step it maintains a <em>partial parse</em>, which is represented as follows:

<ul>
<li>A <em>stack </em>of words that are currently being processed.</li>
<li>A <em>buffer </em>of words yet to be processed.</li>
<li>A list of <em>dependencies </em>predicted by the parser.</li>
</ul>
Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words of the sentence in order. At each step, the parser applies a <em>transition </em>to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied:

<ul>
<li>SHIFT: removes the first word from the buffer and pushes it onto the stack.</li>
<li>LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack.</li>
<li>RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack.</li>
</ul>
On each step, your parser will decide among the three transitions using a neural network classifier.

<ul>
<li>(6 points) Go through the sequence of transitions needed for parsing the sentence <em>“I parsed this sentence correctly”</em>. The dependency tree for the sentence is shown below. At each step, give the configuration of the stack and buffer, as well as what transition was applied this step and what new dependency was added (if any). The first three steps are provided below as an example.</li>
</ul>
<table width="589">
<tbody>
<tr>
<td width="122">Stack</td>
<td width="219">Buffer</td>
<td width="114">New dependency</td>
<td width="135">Transition</td>
</tr>
<tr>
<td width="122">[ROOT]</td>
<td width="219">[I, parsed, this, sentence, correctly]</td>
<td width="114"></td>
<td width="135">Initial Configuration</td>
</tr>
<tr>
<td width="122">[ROOT, I]</td>
<td width="219">[parsed, this, sentence, correctly]</td>
<td width="114"></td>
<td width="135">SHIFT</td>
</tr>
<tr>
<td width="122">[ROOT, I, parsed]</td>
<td width="219">[this, sentence, correctly]</td>
<td width="114"></td>
<td width="135">SHIFT</td>
</tr>
<tr>
<td width="122">[ROOT, parsed]</td>
<td width="219">[this, sentence, correctly]</td>
<td width="114">parsed→I</td>
<td width="135">LEFT-ARC</td>
</tr>
</tbody>
</table>
<ul>
<li>(2 points) A sentence containing <em>n </em>words will be parsed in how many steps (in terms of <em>n</em>)? Briefly explain why.</li>
<li>(6 points) Implement the init and parsestep functions in the PartialParse class in py. This implements the transition mechanics your parser will use. You can run basic (non-exhaustive) tests by running python parsertransitions.py partc.</li>
<li>(6 points) Our network will predict which transition should be applied next to a partial parse. We could use it to parse a single sentence by applying predicted transitions until the parse is complete. However, neural networks run much more efficiently when making predictions about <em>batches </em>of data at a time (i.e., predicting the next transition for any different partial parses simultaneously). We can parse sentences in minibatches with the following algorithm.</li>
</ul>
<strong>Algorithm 1 </strong>Minibatch Dependency Parsing

<strong>Input: </strong>sentences, a list of sentences to be parsed and model, our model that makes parse decisions

Initialize partialparses as a list of PartialParses, one for each sentence in sentences Initialize unfinishedparses as a shallow copy of partialparses <strong>while </strong>unfinishedparses is not empty <strong>do</strong>

Take the first batchsize parses in unfinishedparses as a minibatch

Use the model to predict the next transition for each partial parse in the minibatch

Perform a parse step on each partial parse in the minibatch with its predicted transition

Remove the completed (empty buffer and stack of size 1) parses from unfinishedparses <strong>end while</strong>

<strong>Return: </strong>The dependencies for each (now completed) parse in partialparses.

Implement this algorithm in the minibatchparse function in parsertransitions.py. You can run basic (non-exhaustive) tests by running python parsertransitions.py partd.

<em>Note: You will need </em><em>minibatchparse to be correctly implemented to evaluate the model you will build in part (e). However, you do not need it to train the model, so you should be able to complete most of part (e) even if </em><em>minibatchparse is not implemented yet.</em>

We are now going to train a neural network to predict, given the state of the stack, buffer, and dependencies, which transition should be applied next. First, the model extracts a feature vector representing the current state. We will be using the feature set presented in the original neural dependency parsing paper: <em>A Fast and Accurate Dependency Parser using Neural Networks</em>.<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> The function extracting these features has been implemented for you in utils/parserutils.py. This feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.). They can be represented as a list of integers [<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>m</sub></em>] where <em>m </em>is the number of features and each 0 ≤ <em>w<sub>i </sub>&lt; </em>|<em>V </em>| is the index of a token in the vocabulary (|<em>V </em>| is the vocabulary size). First our network looks up an embedding for each word and concatenates them into a single input vector:

<strong>x </strong>= [<strong>E</strong><em>w</em>1<em>,…,</em><strong>E</strong><em>w</em><em>m</em>] ∈ R<em>dm</em>

where <strong>E </strong>∈ R<sup>|<em>V </em>|×<em>d </em></sup>is an embedding matrix with each row <strong>E</strong><em><sub>w </sub></em>as the vector for a particular word <em>w</em>.

We then compute our prediction as:

<strong>h </strong>= ReLU(<strong>xW </strong>+ <strong>b</strong><sub>1</sub>)

<strong>l </strong>= <strong>hU </strong>+ <strong>b</strong><sub>2</sub>

<strong>y</strong>ˆ = softmax(<em>l</em>)

where <strong>h </strong>is referred to as the hidden layer, <strong>l </strong>is referred to as the logits, <strong>y</strong>ˆ is referred to as the predictions, and ReLU(<em>z</em>) = max(<em>z,</em>0)). We will train the model to minimize cross-entropy loss:

To compute the loss for the training set, we average this <em>J</em>(<em>θ</em>) across all training examples.

<ul>
<li>(10 points) In py you will find skeleton code to implement this simple neural network using PyTorch. Complete the init , embeddinglookup and forward functions to implement the model. Then complete the trainforepoch and train functions within the run.py file.</li>
</ul>
Finally execute python run.py to train your model and compute predictions on test data from Penn Treebank (annotated with Universal Dependencies). Make sure to turn off debug setting by setting debug=False in the main function of run.py.

<strong>Hints:</strong>

<ul>
<li>When debugging, set debug=True in the main function of py. This will cause the code to run over a small subset of the data, so that training the model won’t take as long. Make sure to set debug=False to run the full model once you are done debugging.</li>
<li>When running with debug=True, you should be able to get a loss smaller than 0.2 and a UAS larger than 65 on the dev set (although in rare cases your results may be lower, there is some randomness when training).</li>
<li>It should take about <strong>1 hour </strong>to train the model on the entire the training dataset, i.e., when debug=False.</li>
<li>When running with debug=False, you should be able to get a loss smaller than 0.08 on the train set and an Unlabeled Attachment Score larger than 87 on the dev set. For comparison, the model in the original neural dependency parsing paper gets 92.5 UAS. If you want, you can tweak the hyperparameters for your model (hidden layer size, hyperparameters for Adam, number of epochs, etc.) to improve the performance (but you are not required to do so).</li>
</ul>
<strong>Deliverables:</strong>

<ul>
<li>Working implementation of the neural dependency parser in py. (We’ll look at and run this code for grading).</li>
<li>Report the best UAS your model achieves on the dev set and the UAS it achieves on the test set.</li>
</ul>
<ul>
<li>(12 points) We’d like to look at example dependency parses and understand where parsers like ours might be wrong. For example, in this sentence:</li>
</ul>
Moscow sent troops into Afghanistan . PROPN VERB NOUN ADP PROPN PUNCT the dependency of the phrase <em>into Afghanistan </em>is wrong, because the phrase should modify <em>sent </em>(as in <em>sent into Afghanistan</em>) not <em>troops </em>(because <em>troops into Afghanistan </em>doesn’t make sense). Here is the correct parse:

Moscow&nbsp;&nbsp;&nbsp; sent&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; troops into Afghanistan&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .

PROPN VERB NOUN ADP&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; PROPN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; PUNCT

More generally, here are four types of parsing error:

<ul>
<li><strong>Prepositional Phrase Attachment Error</strong>: In the example above, the phrase <em>into Afghanistan </em>is a prepositional phrase. A Prepositional Phrase Attachment Error is when a prepositional phrase is attached to the wrong head word (in this example, <em>troops </em>is the wrong head word and <em>sent </em>is the correct head word). More examples of prepositional phrases include <em>with a rock</em>, <em>before midnight </em>and <em>under the carpet</em>.</li>
<li><strong>Verb Phrase Attachment Error</strong>: In the sentence <em>Leaving the store unattended, I went outside to watch the parade</em>, the phrase <em>leaving the store unattended </em>is a verb phrase. A Verb Phrase Attachment Error is when a verb phrase is attached to the wrong head word (in this example, the correct head word is <em>went</em>).</li>
<li><strong>Modifier Attachment Error</strong>: In the sentence <em>I am extremely short</em>, the adverb <em>extremely </em>is a modifier of the adjective <em>short</em>. A Modifier Attachment Error is when a modifier is attached to the wrong head word (in this example, the correct head word is <em>short</em>).</li>
<li><strong>Coordination Attachment Error</strong>: In the sentence <em>Would you like brown rice or garlic naan?</em>, the phrases <em>brown rice </em>and <em>garlic naan </em>are both conjuncts and the word <em>or </em>is the coordinating conjunction. The second conjunct (here <em>garlic naan</em>) should be attached to the first conjunct (here <em>brown rice</em>). A Coordination Attachment Error is when the second conjunct is attached to the wrong head word (in this example, the correct head word is <em>rice</em>). Other coordinating conjunctions include <em>and</em>, <em>but </em>and <em>so</em>.</li>
</ul>
In this question are four sentences with dependency parses obtained from a parser. Each sentence has one error, and there is one example of each of the four types above. For each sentence, state the type of error, the incorrect dependency, and the correct dependency. To demonstrate: for the example above, you would write:

<ul>
<li><strong>Error type</strong>: Prepositional Phrase Attachment Error</li>
<li><strong>Incorrect dependency</strong>: troops → Afghanistan</li>
<li><strong>Correct dependency</strong>: sent → Afghanistan</li>
</ul>
<em>Note: There are lots of details and conventions for dependency annotation. If you want to learn more about them, you can look at the UD website: </em><a href="http://universaldependencies.org/"><em>http://universaldependencies.org</em></a><a href="http://universaldependencies.org/"><em>.</em></a><a href="#_ftn5" name="_ftnref5"><em><sup><strong>[5]</strong></sup></em></a><em>However, you </em><em>do not need to know all these details in order to do this question. In each of these cases, we are asking about the attachment of phrases and it should be sufficient to see if they are modifying the correct head. In particular, you </em><em>do not need to look at the labels on the the dependency edges – it suffices to just look at the edges themselves.</em>

i.

I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; was heading&nbsp;&nbsp;&nbsp;&nbsp; to&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; wedding fearing&nbsp;&nbsp;&nbsp;&nbsp; my&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; death&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .

PRON AUX VERB ADP DET NOUN VERB PRON NOUN PUNCT

ii.

iii.

It&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; on&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; loan&nbsp;&nbsp;&nbsp;&nbsp; from&nbsp;&nbsp;&nbsp;&nbsp; a&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; guy&nbsp;&nbsp;&nbsp;&nbsp; named&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Joe&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; O’Neill&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; in&nbsp;&nbsp;&nbsp;&nbsp; Midland&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Texas&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .

<h2>PRON AUX ADP NOUN ADP DET NOUN VERB PROPN PROPN ADP PROPN PUNCT PROPN PUNCT</h2>
iv.

&nbsp;
