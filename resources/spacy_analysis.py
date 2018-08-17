import os
import pandas as pd
from collections import defaultdict, Counter
from IPython.display import display
from random import seed
import random
import math
import csv
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)
# import spacy
# from sussex_nltk.corpus_readers import AmazonReviewCorpusReader
from nltk.corpus import gutenberg
import en_core_web_sm

# from GutenbergCorpus import GutenbergCorpusReader as gcr
#
# reader = gcr.GutenbergCorpusReader()  ## Sussex constructor

nlp = en_core_web_sm.load()
#
emma = gutenberg.raw('austen-emma.txt')
parsed_emma = nlp(emma)
# sense = gutenberg.raw('austen-sense.txt')
# parsed_sense = nlp(sense)
# moby = gutenberg.raw('melville-moby_dick.txt')
# parsed_moby = nlp(moby)

import re

seed(181520)
sample_size = 100
my_sample = random.sample(list(parsed_emma.sents), sample_size)  # select a random sample of sentences
sample = []
for sent in my_sample:
    sent = re.sub("\s+", " ", sent.text)  # clean up the whitespace
    print(sent, "\n")
    sample.append(sent)

entities = []
type_entity = []
sentences = []
for sent in sample:
    parsed_sentence = nlp(sent)
    for ent in parsed_sentence.ents:
        if ent.text not in entities:
            entities.append(ent.text)
            sentences.append(sent)
            type_entity.append(ent.label_)
Entities = pd.DataFrame({'Sentence': sentences, 'Entity': entities, 'Entity_type': type_entity})
print("length Entities:{}, Entities:{}".format(len(Entities), Entities))


## Function to create gender map from names.csv file
def create_gender_map(dict_reader):
    names_info = defaultdict(lambda: {"gender": "", "freq": 0.0})
    for row in input_file:
        name = row["name"].lower()
        if names_info[name]["freq"] < float(row["freq"]):  # is this gender more frequent?
            names_info[name]["gender"] = row["gender"]
            names_info[name]["freq"] = float(row["freq"])
    gender_map = defaultdict(lambda: "unknown")
    for name in names_info:
        gender_map[name] = names_info[name]["gender"]
    return gender_map


input_file = csv.DictReader(open('names.csv'))  ## Importing our names.csv file
gender_map = create_gender_map(input_file)  ## Import the gender map
#### Male homonyms
male_title = ['mr.', 'sir', 'monsieur', 'captain', 'chief', 'master', 'lord', 'baron', 'mister', 'mr', 'prince', 'king']
#### Female homonyms
female_title = ['mrs.', 'ms.', 'miss', 'lady', 'madameoiselle', 'baroness', 'mistress', 'mrs', 'ms', 'queen',
                'princess', 'madam', 'madame']


def gender_guess(name, gender_map):  # Identifying entries in the names.csv database#
    if (len(name.split())) == 1:
        if name.lower() in gender_map.keys():
            return gender_map[name]
        else:
            return 'unknown'

    if (len(name.split())) > 1:
        name_array = name.lower().split()
        if name_array[0] in gender_map.keys():
            return gender_map[name_array[0]]

        for title in name_array:  # Recognising titles of entries#
            if title in male_title:
                return 'male'
            elif title in female_title:
                return 'female'
            else:
                return 'unknown'


def named_entity_counts(document, named_entity_label):
    # Function that outputs a Counter object of human entities found
    occurrences = [ent.string.strip() for ent in document.ents
                   if ent.label_ == named_entity_label and ent.string.strip()]
    return Counter(occurrences)


def Predicts_Gender():
    alice = gutenberg.raw(fileids='carroll-alice.txt')
    parsed_alice = nlp(alice)
    text = parsed_alice  ### Parsing Alice in the wonderland by Lewis Carroll
    entity_type = 'PERSON'  ## Type of entry
    number_of_entities = 10  ### Control over obtaining number of defined type entities
    Entities = pd.DataFrame(named_entity_counts(text, entity_type).most_common(number_of_entities),
                            columns=["Entity", "Count"])
    entity = []
    for char in Entities['Entity']:
        entity.append(gender_guess(char.lower(), gender_map))
    Entities['Pred_Gender'] = entity


def gauss_gender_test():
    # 测试
    names = ['harry', 'abdul', 'homer', 'gary', 'robert', 'wayne', 'lionel']
    for name in names:
        print(gender_guess(name, gender_map))

    names = ['martha', 'holly', 'nicole', 'catherine', 'ruth', 'april', 'christina']
    for name in names:
        print(gender_guess(name, gender_map))

    # #### For first and last names given
    names = ['Liz Lemon', 'Leslie Knope', 'jesus navas', 'Robert Lewandowski', 'Anthony Martial', 'Wesley Sneijder']
    for name in names:
        print(gender_guess(name, gender_map))

    # For names with titles
    print(gender_guess('Sir Alex Ferguson', gender_map))
    print(gender_guess('Lady McElroy', gender_map))
    print(gender_guess('Captain Subash Chandra boBe', gender_map))

    # The program returns 'unknown' if the gender of the entity can't be determined by the function created.
    # This error is a result of the name not being in the names.csv folder or doesn't have a gender
    #  bisecting title attached to it

    print(gender_guess('Liam Neeson', gender_map))
    print(gender_guess('Mahatma Gandhi', gender_map))
    print(gender_guess('Pricella McCartney', gender_map))


# 第二问
from GutenbergCorpus import GutenbergCorpusReader as gcr
from sussex_nltk.corpus_readers import AmazonReviewCorpusReader


def fun1():
    reader = gcr.GutenbergCorpusReader('./data/gutenberg')
    authors = reader.get_authors()

    print('The number of authors mentioned in the Gutenberg corpus are:', format(len(authors)))
    for author in authors:
        print("{0}: {1}".format(author, len(authors[author])))
    # Initialising our problem with parsing the novel by James De Mille for analyzing feature sets of characters.
    #  The name of the selected novel is The Cryptogram
    authors = reader.get_authors()
    tot = 0
    for author in authors:
        tot += len(authors[author])
    works = reader.get_authors_works('De Mille, James')
    for work in works:
        print(work["title"])
    parsed_novel = nlp(works[7]["text"])
    return parsed_novel


"""#### The code returns the feature sets for characters extracted from the novel. The code has added functionality to remove most common occuring verbs in the novel. Also, addition of relations 'nsubj' and 'dobj' are added.


#### Defining our helper functions:
- get_entities_in(parsed_novel,entity_type): inputs parsed novel and entity type to be extracted, returns an array with extracted user defined type entity names

- get_main_characters(parsed_novel,num_characters)
- get_main_characters(parsed_novel,num_characters):
- get_interesting_contexts(novels,rels,num_characters,verb_stop)
- get_pos_in(parsed_novel,pos_type,remove_pos_numb): Extracts all common verbs from the novel and returns a list of all common verbs

remove_pos_numb is the argument that filters common verbs. The number signifies the first n numbers to be removed from the most common verbs. Interseting contexts and characteristics the user has are to be determined by removing most common verbs.

The main function 'get interesting contexts' runs on the parsed novel, entities that are in main characters identified by get_main_characters are identified. Verbs and relations associated with these entities are extracted. 
Stop verbs are removed and a Counter of verbs is returned as output.

"""


def get_entities_in(parsed_novel, entity_type):  ## Get_entities_in returns entity in a novel given the type of entity

    return [ent.text.strip().lower() for ent in parsed_novel.ents
            if ent.label_ == entity_type and ent.text.strip()]


def get_pos_in(parsed_novel, pos_type, remove_pos_numb):
    # Get's the list of desired pos tag in passed tag and filters top n common entries

    verbs = [token.text for token in parsed_novel if token.pos_ == pos_type]
    verbs = [verb.lower() for verb in verbs]
    common_verbs = []
    for row in Counter(verbs).most_common()[:remove_pos_numb]:
        common_verbs.append(row[0])
    return common_verbs


def get_main_characters(parsed_novel, num_characters):
    # Function returns the most commonly occuring characters in a parsed novel

    C = (Counter(get_entities_in(parsed_novel, "PERSON")).most_common()[0:num_characters])
    main_characters = []
    for row in C:
        main_characters.append(row[0])
    return Counter(main_characters)


def get_interesting_contexts(novels, rels, num_characters, verb_stop):
    def of_interest(ent, rels, main_characters):
        return (ent.text.strip().lower() in main_characters
                and ent.label_ == 'PERSON'
                and ent.root.head.pos_ == 'VERB'
                and ent.root.dep_ in rels)

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    return contexts


parsed_novel = ''
novels = {parsed_novel}  ## Parsed novel
number_of_characters_per_text = 8  # Threshold for the number of characters you want contexts for
target_rels = {'nsubj'}  # Relation: Can treat entity as a subject. For object relations enter 'dobj'
target_contexts = get_interesting_contexts(novels, target_rels, number_of_characters_per_text, 1500)
display()  # Display result df
results_df1 = pd.DataFrame.from_dict(target_contexts).applymap(lambda x: '' if math.isnan(x) else x)

C = (Counter(get_entities_in(parsed_novel, "VERB")).most_common())
get_pos_in(parsed_novel, 'VERB', 60)

"""#### Since we are interested in quantifying the extent to which authors exhibit gender-based distinctions in the way they in the way that they portray their main characters, it would be useful to base this not only on the contexts of places where a character is mentioned by name, but also when a character is mentioned with a pronoun. The pronouns "he", "she", "his" and "her" indicate the gender of the person being referred to, so provide a reliable source of additional data.

#### The following code cell shows how these pronouns can be extracted from a text using the `noun_chunks` property of a parsed document. Noun chunks are "base noun phrases" – flat phrases that have a noun as their head. You can think of noun chunks as a noun plus the words describing the noun – for example, "the lavish green grass" or "the world’s largest tech fund". 


#### Extracting features from pronouns in the parsed_novel, there were 95 of these features found, after removing commonly occuring verbs. The function gendered_pronoun takes in an argument as the noun phrase and returns the noun phrases that have pronouns ;he, she, is and her'. These pronoun phrases are then used to extract the associative word attached to the pronoun. Common verbs occuring in the text are removed by the use of the previously created get_pos_in function. 

"""


def gendered_pronoun(np):
    return np.text.strip() in ["he", "she", "her", "his"]


stop_verbs = get_pos_in(parsed_novel, 'VERB', 1400)
text = parsed_novel
nounphrases = [[re.sub("\s+", " ", np.text), np.root.head.text] for np in parsed_novel.noun_chunks if
               gendered_pronoun(np) and np.root.head.text not in stop_verbs]
print("There were {} noun phrases found.".format(len(nounphrases)))
df = (pd.DataFrame(nounphrases, columns=['Pronoun', 'Verb']))

# Fetch feature sets of the 2 novels parsed of all the male characters
works = reader.get_authors_works('Holmes, Mary Jane')
parsed_novel1 = nlp(works[2]["text"])
works = reader.get_authors_works('Wells, H. G. (Herbert George)')
parsed_novel2 = nlp(works[10]["text"])


def get_interesting_contexts_gender(novels, rels, num_characters, verb_stop, gender):
    def of_interest(ent, rels, main_characters, gender):
        if gender_guess(ent.text.strip().lower(), gender_map) == gender:
            return (ent.text.strip().lower() in main_characters
                    and ent.label_ == 'PERSON'
                    and ent.root.head.pos_ == 'VERB'
                    and ent.root.dep_ in rels)

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters, gender):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    return contexts


novels = {parsed_novel1, parsed_novel2}
number_of_characters_per_text = 8
target_rels = {'nsubj', 'dobj'}
verb_stop = 1000
target_contexts = get_interesting_contexts_gender(novels, target_rels, number_of_characters_per_text, verb_stop, 'male')
# display()
results_df2 = pd.DataFrame.from_dict(target_contexts).applymap(lambda x: '' if math.isnan(x) else x)

C = (Counter(get_entities_in(parsed_novel, "VERB")).most_common())
get_pos_in(parsed_novel, 'VERB', 60)


# Fetch feature sets of the 2 novels parsed of all the female characters.

def get_interesting_contexts_gender(novels, rels, num_characters, verb_stop, gender):
    def of_interest(ent, rels, main_characters, gender):
        if gender_guess(ent.text.strip().lower(), gender_map) == gender:
            return (ent.text.strip().lower() in main_characters
                    and ent.label_ == 'PERSON'
                    and ent.root.head.pos_ == 'VERB'
                    and ent.root.dep_ in rels)

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters, gender):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    return contexts


novels = {parsed_novel1, parsed_novel2}
number_of_characters_per_text = 7
target_rels = {'nsubj', 'dobj'}
verb_stop = 1000
target_contexts = get_interesting_contexts_gender(novels, target_rels, number_of_characters_per_text, verb_stop,
                                                  'female')
display(pd.DataFrame.from_dict(target_contexts).applymap(lambda x: '' if math.isnan(x) else x))
# C=(Counter(get_entities_in(parsed_novel,"VERB")).most_common())
# get_pos_in(parsed_novel,'VERB',60)


from sklearn.metrics.pairwise import cosine_similarity


def get_interesting_contexts_gender(novels, rels, num_characters, verb_stop, gender):
    def of_interest(ent, rels, main_characters, gender):
        if gender_guess(ent.text.strip().lower(), gender_map) == gender:
            return (ent.text.strip().lower() in main_characters
                    and ent.label_ == 'PERSON'
                    and ent.root.head.pos_ == 'VERB'
                    and ent.root.dep_ in rels)

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters, gender):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    df = (pd.DataFrame.from_dict(contexts).applymap(lambda x: 0 if math.isnan(x) else x))
    return Counter(dict(df.sum(axis=1)))


A = get_interesting_contexts_gender(novels, target_rels, number_of_characters_per_text, verb_stop, 'male')
B = get_interesting_contexts_gender(novels, target_rels, number_of_characters_per_text, verb_stop, 'female')


def counters_to_feature_lists(counter1, counter2):
    combined = counter1 + counter2
    list1 = [counter1[key] for key in combined]
    list2 = [counter2[key] for key in combined]
    return list1, list2


L1, L2 = counters_to_feature_lists(A, B)
print(L1)
print(L2)
res = cosine_similarity([L1], [L2])[0, 0]
print(res)


def get_interesting_contexts(novels, rels, num_characters, verb_stop):
    def of_interest(ent, rels, main_characters):
        return (ent.text.strip().lower() in main_characters
                and ent.label_ == 'PERSON'
                and ent.root.head.pos_ == 'VERB'
                and ent.root.dep_ in rels)

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    df = (pd.DataFrame.from_dict(contexts).applymap(lambda x: 0 if math.isnan(x) else x))
    A = df.iloc[:, :5]
    A_ = Counter(dict(A.sum(axis=1)))
    #     B=df.iloc[:,5:]
    #     A_=Counter(dict(B.sum(axis=1)))
    return A_


novels = {parsed_novel1, parsed_novel2}
number_of_characters_per_text = 8
target_rels = {'nsubj', 'dobj'}
A_count = get_interesting_contexts(novels, target_rels, number_of_characters_per_text, 100)


def get_interesting_contexts(novels, rels, num_characters, verb_stop):
    def of_interest(ent, rels, main_characters):
        return (ent.text.strip().lower() in main_characters
                and ent.label_ == 'PERSON'
                and ent.root.head.pos_ == 'VERB'
                and ent.root.dep_ in rels)

    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    df = (pd.DataFrame.from_dict(contexts).applymap(lambda x: 0 if math.isnan(x) else x))
    B = df.iloc[:, 5:]
    B_ = Counter(dict(B.sum(axis=1)))
    #     B=df.iloc[:,5:]
    #     A_=Counter(dict(B.sum(axis=1)))
    return B_


B_count = get_interesting_contexts(novels, target_rels, number_of_characters_per_text, 100)


def counters_to_feature_lists(counter1, counter2):
    combined = counter1 + counter2
    list1 = [counter1[key] for key in combined]
    list2 = [counter2[key] for key in combined]
    return list1, list2


L1, L2 = counters_to_feature_lists(A_count, B_count)
print(L1)
print(L2)
a = cosine_similarity([L1], [L2])[0, 0]
print(a)
