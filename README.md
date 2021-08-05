# SNLP project 

Name 1: Minh Tung Phung<br/>
Id 1: 7012522<br/>
Email id 1: miph00001@teams.uni-saarland.de<br/>


Name 2: Pablo Valdunciel<br/>
Id 2: 7010186 <br/>
Email id 2: pava00001@stud.uni-saarland.de<br/> 




## A. English 

### 1. Data Preparation - Preprocessing 

**Assumptions**:
- Two-or-more end-of-line symbols (i.e. '\n\n') signify the separation of two paragraphs.
- No single sentence resides in more than one paragraph.
- We are allowed to use nltk for sentence-tokenization. The function from nltk (nltk.tokenize.sent_tokenize) is sufficient for the task.
- There is no situation where double-space (i.e. '  ') makes sense, so we contract all sequences of more than one space into a space character (i.e. from the regex point of view, ' +' is substituted with ' ').

**Operations**:
- Get a list of paragraphs by splitting the text by at-least-2 consecutive end-of-line characters (i.e. '\n{2,}').
- For each paragraph:
    + Replace all end-of-line characters with a space.
    + Contract all sequences of more than one space into a space character.
    + Tokenizer into sentences.
- Collect all sentences from all paragraphs as a list and return.

### 2. Subword Segmentation

#### a. Experiments with vocabulary sizes

There are 2 types of subword: small vocabulary size (i.e. usually from 100 to 800) and large vocabulary size (i.e. from 1500 to 3000). For each type, we do some experiments to choose an exact vocabulary size, which will be fixed for subsequent tasks.

**Approach**:

We propose to employ the [Minimum Description Length (MDL)](https://en.wikipedia.org/wiki/Minimum_description_length) principle to tackle this task. The MDL is a technique for model selection inspired by Occam's razor. In its basic form, MDL prefers the model which summarizes the data the best, i.e. the total size of the model and the description of the data given the model is the smallest.

$$
L = L(M) + L(D|M)
$$

For our current task, we formalize the description length as the total size of the model file, the vocab file and the encoded text file. In which, the model file and the vocab file represent the model, and the encoded file represents the data given the model.

$$
\begin{align*}
&L(M) = \text{size of .model file + size of .vocab file (in bytes),} \\
&L(D|M) = \text{size of encoded file.}
\end{align*}
$$

To prevent from overfitting, we only consider vocab sizes rounded to hundredth, i.e. 100, 200, ..., 800 and 1500, 1600, ..., 3000. The best vocabulary size is chosen as the one which results in the smallest description length per the formulation above.

**Experiment details**:

a. Small vocabulary:

For small vocabulary sizes (from 100 to 800), the corresponding description length (or total file size) is shown in the figure below. The smallest description length belongs to the vocabulary size of 500.

![figures/Task2.3.small_vocab_file_size.png](figures/Task2.3.small_vocab_file_size.png)

One interesting observation is that the description length decreases sharply from vocabulary size of 100 to 200. For a qualitative evalution, let us look at an examplar sentence:

original text:

> There was a table set out under a tree in front of the house, and the March Hare and the Hatter were having tea at it: a Dormouse was sitting between them, fast asleep, and the other two were using it as a cushion, resting their elbows on it, and talking over its head.

vocab_size = 100:

> ▁ T he re ▁w a s ▁a ▁t a b l e ▁s e t ▁ ou t ▁ u nd er ▁a ▁t re e ▁ in ▁ f r o n t ▁o f ▁the ▁ h ou s e , ▁and ▁the ▁ M a r c h ▁ H a re ▁and ▁the ▁ H a t t er ▁w er e ▁ ha v ing ▁t e a ▁a t ▁ it : ▁a ▁ D o r m ou s e ▁w a s ▁s it t ing ▁b e t w e en ▁the m , ▁ f a s t ▁a s l e e p , ▁and ▁the ▁o t he r ▁t w o ▁w er e ▁ u s ing ▁ it ▁a s ▁a ▁c u s h i o n , ▁ re s t ing ▁the i r ▁ e l b o w s ▁o n ▁ it , ▁and ▁t a l k ing ▁o v er ▁ it s ▁ he a d .

vocab_size = 200:

> ▁T he re ▁was ▁a ▁t a b le ▁s et ▁ out ▁u nd er ▁a ▁t re e ▁in ▁f r on t ▁of ▁the ▁h ou se , ▁and ▁the ▁M ar ch ▁H a re ▁and ▁the ▁H at ter ▁w er e ▁ha v ing ▁t ea ▁at ▁it : ▁a ▁D or m ou se ▁was ▁s it t ing ▁be t w e en ▁the m , ▁f as t ▁as le e p , ▁and ▁the ▁o t her ▁t w o ▁w er e ▁u s ing ▁it ▁as ▁a ▁c u s h i on , ▁re st ing ▁the ir ▁e l b ow s ▁on ▁it , ▁and ▁t a l k ing ▁o ver ▁it s ▁he ad .

vocab_size = 500:

> ▁The re ▁was ▁a ▁t able ▁s et ▁out ▁u nder ▁a ▁t ree ▁in ▁fr on t ▁of ▁the ▁h ouse , ▁and ▁the ▁March ▁H are ▁and ▁the ▁Hatter ▁were ▁ha v ing ▁t ea ▁at ▁it : ▁a ▁Dormouse ▁was ▁s it ting ▁be t w een ▁them , ▁f ast ▁as le e p , ▁and ▁the ▁other ▁tw o ▁were ▁us ing ▁it ▁as ▁a ▁c us h ion , ▁re st ing ▁their ▁e l b ow s ▁on ▁it , ▁and ▁t al king ▁over ▁its ▁head .

For the size of 100, nearly every token is a character, which is obviously a bad sign since it cannot utilize the use of longer subwords or words to foster learning. Note that since this text has roughtly 72 characters, only 100-72=28 combinations of characters are taken as tokens for this size. 

For the size of 200, more meaningful combinations of characters are recognized as a token. In particular, the number almost quintuples from 28 to 128. As an example, *'ing'* has its own place. This explains the great drop in the description length.

This trend continues up to the size of 500, where the trade-off between generality and specificity of adding new tokens equalizes. Most of the stopwords, which are very frequent, occupy their own tokens. Meaningful suffixes like *'able'* now also have their own place. 

While a vocabulary size of 500 is quite small in practice, we believe it is suitable for our current problem since the input text file is small.

b. Large vocabulary

For large vocabulary sizes (from 1500 to 3000), we apply the same procedure as described. The description lengths corresponding to different sizes are shown below: 

![figures/Task2.3.large_vocab_file_size.png](figures/Task2.3.large_vocab_file_size.png)

It can be observed from the figure that the larger the vocabulary size, the longer the description length. We suspect that this is because 1500 is already bigger than the optimal size, thus adding more tokens only makes things worse. For the purposes of the next tasks, we will use the size of 1500 for the large vocabulary size.

#### b. Training of Segmentation models

**Operations**:
- We train 3 segmentation models of different vocabulary sizes:
    - character-level segmentation.
    - small subword vocabulary size of 500.
    - large subword vocabulary size of 1500.
- For each vocabulary size, we:
    - train a segmentation model.
    - apply the model on the training data to get an encoded text.
    - decode the encoded text, verify that the decoded text is the same as the original training data to ensure the correctness of the model.
    - apply the same encoding, decoding operations on the test data.    

## B. Bengali

### 1. Data Preparation - Preprocessing 

**Assumptions:**
- Two-or-more end-of-line symbols (i.e. '\n\n') signify the separation of two paragraphs.
- Two or more exclamation(!)/interrogation(?)/full stop(।) doesn't follow the grammatical rules; it is 
    most likely used by the writer to emphasize whatever they are saying.
- The end of a sentence is determined by an exclamation (!), an interrogation (?) or a full stop (।)
- The corpus is probably from a web page, since it includes HTML labels
- Presence of text in English

**Operations**: 
- Get a list of paragraphs by splitting the text by at-least-2 consecutive end-of-line characters (i.e. '\n{2,}').
- For each paragraph:
    + Substitute two or more exclamations/interrogations/full stops by a single one
    + Remove HTML tags
    + Remove text in English
    + Split text in sentences using the [bltk library](https://github.com/saimoncse19/bltk)
- Collect all sentences from all paragraphs as a list and return.