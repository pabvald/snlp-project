# SNLP project 

Name 1: Minh Tung Phung<br/>
Id 1: 7012522<br/>
Email id 1: miph00001@teams.uni-saarland.de<br/>


Name 2: Pablo Valdunciel<br/>
Id 2: 7010186 <br/>
Email id 2: pava00001@stud.uni-saarland.de<br/> 


## 1. Data Preparation - Preprocessing 

### English 

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

### Bengali
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