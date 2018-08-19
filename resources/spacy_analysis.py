import re
import pandas as pd
from collections import defaultdict, Counter
from random import seed
import random
import math
import csv
import en_core_web_sm
from nltk.corpus import gutenberg


def get_austen_emma_sample():
    nlp = en_core_web_sm.load()
    emma = gutenberg.raw('austen-emma.txt')
    parsed_emma = nlp(emma)
    seed(181520)
    sample_size = 100
    my_sample = random.sample(list(parsed_emma.sents), sample_size)
    sample = []
    for sent in my_sample:
        sent = re.sub("\s+", " ", sent.text)
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
    return Entities


## Function to create gender map from names.csv file
def create_gender_map(input_file):
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


def named_entity_counts(document, named_entity_label):
    ## Function that outputs a Counter object of human entities found
    occurrences = [ent.string.strip() for ent in document.ents
                   if ent.label_ == named_entity_label and ent.string.strip()]
    return Counter(occurrences)


def gender_guess(name, gender_map):
    male_title = ['mr.', 'sir', 'monsieur', 'captain', 'chief', 'master', 'lord', 'baron', 'mister', 'mr', 'prince',
                  'king']
    #### Female homonyms
    female_title = ['mrs.', 'ms.', 'miss', 'lady', 'madameoiselle', 'baroness', 'mistress', 'mrs', 'ms', 'queen',
                    'princess', 'madam', 'madame']

    if (len(name.split())) == 1:
        if name.lower() in gender_map.keys():
            return gender_map[name]
        else:
            return ('unknown')

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


def gender_classifier():
    nlp = en_core_web_sm.load()

    input_file = csv.DictReader(open('names.csv'))
    gender_map = create_gender_map(input_file)
    #### Male homonyms

    alice = gutenberg.raw(fileids='carroll-alice.txt')
    parsed_alice = nlp(alice)
    entity_type = 'PERSON'
    number_of_entities = 10
    Entities = pd.DataFrame(named_entity_counts(parsed_alice, entity_type).most_common(number_of_entities),
                            columns=["Entity", "Count"])
    entity = []
    for char in Entities['Entity']:
        entity.append(gender_guess(char.lower(), gender_map))
    Entities['Pred_Gender'] = entity
    return Entities


def gender_classifierTest():
    input_file = csv.DictReader(open('names.csv'))
    gender_map = create_gender_map(input_file)
    names = ['harry', 'abdul', 'homer', 'gary', 'robert', 'wayne', 'lionel']
    for name in names:
        print(gender_guess(name, gender_map))
    names = ['martha', 'holly', 'nicole', 'catherine', 'ruth', 'april', 'christina']
    for name in names:
        print(gender_guess(name, gender_map))

    names = ['Liz Lemon', 'Leslie Knope', 'jesus navas', 'Robert Lewandowski', 'Anthony Martial', 'Wesley Sneijder']
    for name in names:
        print(gender_guess(name, gender_map))

    print(gender_guess('Sir Alex Ferguson', gender_map))
    print(gender_guess('Lady McElroy', gender_map))
    print(gender_guess('Captain Subash Chandra boBe', gender_map))

    print(gender_guess('Liam Neeson', gender_map))
    print(gender_guess('Mahatma Gandhi', gender_map))
    print(gender_guess('Pricella McCartney', gender_map))


#  Topic 8

def get_entities_in(parsed_novel, entity_type):
    return [ent.text.strip().lower() for ent in parsed_novel.ents
            if ent.label_ == entity_type and ent.text.strip()]


def get_pos_in(parsed_novel, pos_type, remove_pos_numb):
    ## Get's the list of desired pos tag in passed tag and filters top n common entries

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


def of_interest(ent, rels, main_characters):
    print(ent.root.dep_ , '158')
    return (ent.text.strip().lower() in main_characters
            and ent.label_ == 'PERSON'
            and ent.root.head.pos_ == 'VERB'
            and ent.root.dep_ in rels)


def get_interesting_contexts(novels, rels, num_characters, verb_stop):
    contexts = defaultdict(Counter)
    for parsed_novel in novels:
        main_characters = get_main_characters(parsed_novel, num_characters)
        stop_verbs = get_pos_in(parsed_novel, 'VERB', verb_stop)
        for ent in parsed_novel.ents:
            if of_interest(ent, rels, main_characters):
                if ent.root.head.lemma_ not in stop_verbs:
                    contexts[ent.text.strip().lower()][ent.root.head.lemma_] += 1
    return contexts


def extracted_features_for_characters(parsed_novel):
    novels = {parsed_novel}
    number_of_characters_per_text = 8
    target_rels = {'nsubj'}
    target_contexts = get_interesting_contexts(novels, target_rels, number_of_characters_per_text, 1500)
    results = pd.DataFrame.from_dict(target_contexts).applymap(lambda x: '' if math.isnan(x) else x)
    return results


def gendered_pronoun(np):
    return np.text.strip() in ["he", "she", "her", "his"]


def get_nounphrases(parsed_novel):
    stop_verbs = get_pos_in(parsed_novel, 'VERB', 1400)
    nounphrases = [[re.sub("\s+", " ", np.text), np.root.head.text] for np in parsed_novel.noun_chunks if
                   gendered_pronoun(np) and np.root.head.text not in stop_verbs]
    print("There were {} noun phrases found.".format(len(nounphrases)))
    df = (pd.DataFrame(nounphrases, columns=['Pronoun', 'Verb']))
    return df


if __name__ == '__main__':

    method = 'extracted_features_for_characters'

    if method == 'gender_classifier':
        df = gender_classifier()
        df.to_csv("gender_classifier.csv", index=None)

    if method == 'gender_classifierTest':
        gender_classifierTest()

    if method == 'extracted_features_for_characters':
        nlp = en_core_web_sm.load()

        alice = gutenberg.raw(fileids='carroll-alice.txt')
        parsed_alice = nlp(alice)
        df = extracted_features_for_characters(parsed_alice)
        df.to_csv("extracted_features_for_characters.csv", index=None)

    if method == 'get_nounphrases':
        nlp = en_core_web_sm.load()
        alice = gutenberg.raw(fileids='carroll-alice.txt')
        parsed_alice = nlp(alice)
        df = get_nounphrases(parsed_alice)
        df.to_csv("get_nounphrases.csv", index=None)
