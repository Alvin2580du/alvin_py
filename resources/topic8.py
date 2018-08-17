import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
from itertools import zip_longest
from IPython.display import display
from random import seed
import random
import math
from pylab import rcParams
from operator import itemgetter, attrgetter, methodcaller
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import csv
from operator import itemgetter, attrgetter, methodcaller
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)
import spacy
from sussex_nltk.corpus_readers import AmazonReviewCorpusReader
from nltk.corpus import gutenberg
import en_core_web_sm
nlp = en_core_web_sm.load()

from GutenbergCorpus import GutenbergCorpusReader as gcr
reader = gcr.GutenbergCorpusReader()                         ## Sussex constructor


"""## Overview
In this topic you will be using spaCy's named entity extractor and the gender classifier that you created in Topic 7 to characterise differences in the way that an author portrays male and female characters.

We will look at how it is possible to capture apspects of the way in which characters are portrayed, in terms of features. Each character in a novel will be represented in terms of a **feature set**. For example, one option is that the features are the verbs that the character is the object of (giving a rough sense of what the character does).

For each character, we will collect a set of features and represent the feature set associated with a character as a special kind of dictionary called a `Counter`. Each feature is used as a key and the counter maps that feature to a weight which could, for example, be a count indicating how many times that feature has been seen.

Given that we have a way to guess the gender of some characters, we can aggregate feature sets across all characters of a given gender. Indeed, we can aggregate male and female feature sets across all novels written by a given author or set of authors.

Once we have done this we will look at how to measure the similarity of the resulting (aggregated) feature sets.

First, however, we look at how you can gain access to the texts of a substantial collection of novels.

"""

"""### Gutenberg electronic text archive
[Project Gutenberg electronic text archive](http://www.gutenberg.org/) contains around 75,000 free electronic books. We have made 14862 of the text available to you. 


To access these texts run the following cell.

If you are working on your own machine you will need to do the following:
- download and unzip the file `\\ad.susx.ac.uk\ITS\TeachingResources\Departments\Informatics\LanguageEngineering\resources.zip'`
- update the directory in the following cell, changing the string
`'\\ad.susx.ac.uk\ITS\TeachingResources\Departments\Informatics\LanguageEngineering\resources\data\gutenberg_eng'` to be the location of the directory within the resources folder that contains "authors.p" and "cleaned_meta_gutenberg"

"""

# Get a pickled dictionary of the authors in our extended Gutenberg collection
# Key = author name - string which is used to index and retrieve the works of the author
# Value = The names of the files containing the works of that author
authors = reader.get_authors()

# Let's find out how many texts we have in total.

tot = 0
for author in authors:
    tot += len(authors[author])

print("The collection contains text written by {} different authors".format(len(authors)))
print("There are a total of {} texts in the collection".format(tot))


"""### Exercise
Run the following cell to see a list of all of the authors with texts in the collection and the number of texts for each author.

"""

for author in authors:
    print("{0}: {1}".format(author,len(authors[author])))

exit(1)
"""### Obtaining the text of a novel

The Gutenberg Corpus Reader class provides a method, `get_authors_works`, that returns all of the works of a specified author.

If you run
```
works = reader.get_authors_works(<AUTHOR NAME>)
```
`works` will be a list of dictionaries where each dictionary in the list is one of the works written by the specified author.
- each dictionary in the list has three keys: 
 - "author" that maps to the name of the author
 - "title" that maps to the title of the text
 - "text" that maps to the raw text of the text
 
### Exericse

- Choose one of the authors listed when you run the cell above.
- Adapt the following cell to see the titles of the works available for your chosen author in our collection.


"""


works = reader.get_authors_works()  #  replace <AUTHOR NAME> by a string that is the name of an author
for work in works:
    print(work["title"])


"""### Exercise
- In the blank code cell below run spacy on the texts of one of the novels by of your chosen author.

Later you will be exploring a collection of novels, but for now, it is sufficient to work with a single novel.

"""

parsed_Middlemarch = nlp(works[3]["text"])


"""### Exercise

In the blank cell below, define a function `get_entities_in(parsed_novel,entity_type)` that takes two inputs:
- `parsed_novel` is the result of running spaCy on the raw text of some novel
- `entity_type` is one of the spaCy entity types, e.g. "PERSON"

The output should be a list of the text for each entity appearing in `parsed_novel` that is of type `entity_type`

spaCy can sometimes return entities with an empty text representation, and you don't want to include these in the output.

It is helpful to normalise the text as follows:
- convert the text for each entity to lower case using `lower()`
- remove any surrounding white space, using `strip()`

Run your function on your parsed novel and look at the first 10 characters.


"""


"""### Getting the main characters from a novel

Your next talks is to define a function `get_main_characters(parsed_novel,num_charachters)` that takes two inputs:
- `parsed_novel` is the result of running spaCy on the raw text of some novel
- `num_charachters` is a positive whole number, specifying how many of the main characters should be returned

The output will be a list of the `num_characters` most frequently occurring `"PERSON"` entities in `parsed_novel`.

### Exercise
In the blank cell below, implement `get_main_characters`.
- This function should make use of the `get_entities` function you have just defined
- You can use `Counter` to produce a counter from a list of elements - try `Counter(["a","b","a","c","b"])`
- Once you have a `Counter` you can use `Counter`'s `most_common` method to find the most comment characters

"""

"""### Extracting Feature Sets for Characters

We now turn to the issue of extracting feature sets for characters or sets of characters.

As explained above, we will store each` feature sets as a `Counter`

"""

"""### Exercise
- Examine the following code cell and see if you can work out what it is doing.
- Edit the code so that the novel you are working with is being used
- Run the cell and look at the output to establish if your understanding is correct.

"""


def get_interesting_contexts(novels, num_characters):
    def of_interest(ent, main_characters):
        return (ent.text.strip().lower() in main_characters
                and ent.label_ == 'PERSON'
                and ent.root.head.pos_ == 'VERB')

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        for ent in parsed_novel.ents:
            if of_interest(ent, main_characters):
                contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    return contexts


# novels = { < PARSED_NOVEL >}  # use a set here to allow for the possibility of having multiple texts
number_of_characters_per_text = 8
target_contexts = get_interesting_contexts(novels, number_of_characters_per_text)
df1  = pd.DataFrame.from_dict(target_contexts).applymap(lambda x: '' if math.isnan(x) else x)
print(df1)



"""### Exercise
Make a copy of the code cell above and adapt the code so that it only counts situations where the person is the subject of the verb, 
i.e. in an `nsubj` relation. This identifies the things that the person does. 
 
- write your code so that it is possible to specify any set of relations of interest, e.g. both `nsubj` and `dobj`
- run versions of your code for both `nsubj` and `dobj`, the latter revealing things that are done to the person.

"""


"""### Exercise
Refine your solution futher by removing the most commonly occurring verbs.
Adapt a copy of the code that you have created when solving the previous exercise so that contexts involving the most  common verbs are not displayed. 

Hint: use a `Counter` to determine the count of each verb in a set of novels, and then use `most_common(n)` to find the most common n verbs.


"""

"""### Exercise
Spend some time further refining your solution. Your goal shoudl be to indentify other aspects of the context where a character 
is mentioned that you think will help to provide a richer characterisation of the way that a character is being portrayed by the author.


"""

"""### Aggregating feature sets

Once you are satisifed with the feature sets that you are able to build for a character, you are ready to undertake your analysis 
of the way characters are being portrayed based on gender.

- Select a set of novels
- Parse each of the novels with spaCy (this might take a while)
- Determine the settings of any parameters that are needed by the code you have written to produce the character feature sets, e.g.
 - the number characters to consider in each novel
 - the number of most common verbs to disregard
- Run your code that builds feature sets for characters over all of the novels under consideration
- Build two aggregated feature sets, one for all female characters and one for all male characters

In the next cell, we look at how to measure the difference between these two aggregated feature sets and how to assess whether the different you find is significant.

"""



"""### Measuring the similarity of two feature sets

The code cell below shows how to compare the similarity of two feature sets. This is now explained.

- We are given two feature sets: `A` and `B`.
- Initially, each feature set is represented as a `Counter` which is a dictionary where the keys are the features and each feature (key) is mapped to a positive number which corresponds to the strength (weight) of that feature. 
 - feature set `A` has features `'a', 'b' and 'c'` with weights `1, 2 and 3`, respectively.
 - feature set `B` has features `'b', 'c', 'd' and 'e'` with weights `3, 4, 5 and 6`, respectively.
- Note that they share some, but not all of their features.
- Our goal is to represent both feature sets as lists in such a way that each position in a lists is consistently used for a particular feature
- For example, we could use a list with 5 positions, where the weight of feature `'a'` is held in the first position, the weight of feature `'b'` is held in the second position, and so on. 
 - with this scheme the feature list for `A` would be the list: `[1,2,3,0,0]`, and the feature list for `B` would be `[0,3,4,5,6]`.
- The function `counters_to_feature_lists` takes two feature sets each of which is a `Counter` and returns two lists, one for each of the inputs, where both lists use the same feature representation.
- In the first line of the function, the counters are added together. This is done because the keys of resulting counter (which is named `combined`) can be used to produce consistent mappings of the counters to lists - see lines 2 and 3.
- Once consistent list representations are produced for the two feature sets, we can use the `cosine_similarity` function from `sklearn` as as a measure of how similar the lists are, and therefore, how similar the feature sets are.
- `cosine_similarity` returns a real number between 0 and 1, with 1 indicating that the inputs are identical, and 0 indicating that the two inputs are completely different.


"""

from sklearn.metrics.pairwise import cosine_similarity

A = Counter({'a':1, 'b':2, 'c':3})
B = Counter({'b':3, 'c':4, 'd':5, 'e':6})

def counters_to_feature_lists(counter1,counter2):
    combined = counter1 + counter2
    list1 = [counter1[key] for key in combined]
    list2 = [counter2[key] for key in combined]
    return list1,list2

L1,L2 = counters_to_feature_lists(A,B)
print(L1)
print(L2)
res = cosine_similarity([L1], [L2])[0,0]
print("cosine_similarity:{}".format(cosine_similarity))


"""### When is a difference a significant difference?

The male and female feature sets that you have produced will not be identical, so will have a cosine similarity of less than one.

In order to assess whether there is strong evidence that males and females are portrayed differently in the novels you have chosen, you need to compare this cosine similarity with random non-gender based splits of the characters.

In order to do this, create a random gender classifier and undertake the same analysis with this as above to produce a cosine similarity. By repeating this process several times you will get a sense of how much variation in cosine similiarity is found when doing this.

Another consideration that should be considered is that low cosine similarity values might result from a large difference in the number of male and female characters. To check this, repeat the above process,  making sure that you use feature sets from exactly the same number of male and female characters.


"""



"""### Extracting Gendered Pronouns 

Since we are interested in quantifying the extent to which authors exhibit gender-based distinctions in the way they in the way that they portray their main characters, it would be useful to base this not only on the contexts of places where a character is mentioned by name, but also when a character is mentioned with a pronoun. The pronouns "he", "she", "his" and "her" indicate the gender of the person being referred to, so provide a reliable source of additional data.

The following code cell shows how these pronouns can be extracted from a text using the `noun_chunks` property of a parsed document.


"""


def gendered_pronoun(np):
    return np.text.strip() in ["he", "she", "her", "his"]

text = parsed_emma
nounphrases = [[re.sub("\s+"," ",np.text), np.root.head.text] for np in parsed_emma.noun_chunks if gendered_pronoun(np)]
print("There were {} noun phrases found.".format(len(nounphrases)))
df2 = pd.DataFrame(nounphrases)
print(df2)



