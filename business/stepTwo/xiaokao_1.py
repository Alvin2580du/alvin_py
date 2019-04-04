import collections
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np

SUIT_LIST = ("Hearts", "Spades", "Diamonds", "Clubs")
NUMERAL_LIST = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace")


class card:
    def __init__(self, numeral, suit):
        self.numeral = numeral
        self.suit = suit
        self.card = self.numeral, self.suit

    def __repr__(self):
        return self.numeral + "−" + self.suit


class poker_hand():
    def __init__(self, cardlist):
        self.card_list = cardlist

    def __repr__(self):
        short_desc = "Nothing"
        numeral_dict = collections.defaultdict(int)
        suit_dict = collections.defaultdict(int)
        for my_card in self.card_list:
            numeral_dict[my_card.numeral] += 1
            suit_dict[my_card.suit] += 1
        if len(numeral_dict) == 4:
            short_desc = 'One_pair'
        elif len(numeral_dict) == 3:
            if 3 in numeral_dict.values():
                short_desc = 'Three-of-a-kind'
            else:
                short_desc = 'Two-pair'

        elif len(numeral_dict) == 2:
            if 2 in numeral_dict.values():
                short_desc = 'Full_house'
            else:
                short_desc = 'Four-of-a-kind'
        else:
            stright, flush = False, False

            if len(suit_dict) == 1:
                flush = True
                min_numeral = min([NUMERAL_LIST.index(x) for x in numeral_dict.keys()])
                max_numeral = min([NUMERAL_LIST.index(x) for x in numeral_dict.keys()])

                if int(min_numeral) - int(max_numeral) == 4:
                    straight = True
                    low_straight = set(("Ace", "2", "3", "4", "5"))
                    if not set(numeral_dict.keys()).difference(low_straight):
                        straight = True
                    if straight and not flush:
                        short_desc = 'Straight'
                    elif flush and not straight:
                        short_desc = 'Flush'
                    elif flush and straight:
                        short_desc = 'Straight_Flush'
        enumeration = "/".join([str(x) for x in self.card_list])
        return "{enumeration}_{short_desc}".format(**locals())


class deck(set):
    def __init__(self):
        for numeral, suit in itertools.product(NUMERAL_LIST, SUIT_LIST):
            self.add(card(numeral, suit=suit))

    def get_card(self):
        a_card = random.sample(self, 1)[0]
        self.remove(a_card)
        return a_card

    def get_hand(self, number_of_cards=5):
        if number_of_cards == 5:
            tmp = [self.get_card() for x in range(number_of_cards)]
            return poker_hand(tmp)
        elif number_of_cards == 3:
            tmp = [card('Ace', suit='Spades'), card('Ace', suit='Spades')] + [self.get_card() for x in range(number_of_cards)]
            return poker_hand(tmp)
        else:
            raise NotImplementedError


def build_one(step='one', number_of_hands=250000):
    print("--------------Question:{}-------------------".format(step))
    Straight_Flush = 0
    Four_of_a_Kind = 0
    Full_House = 0
    Flush = 0
    Straight = 0
    Three_of_a_Kind = 0
    Two_Pair = 0
    One_pair = 0
    Nothing = 0
    error_num = 0
    for i in range(number_of_hands):
        try:
            if step == 'one':
                results = deck().get_hand(number_of_cards=5)
            else:
                results = deck().get_hand(number_of_cards=3)

            if str(results).split("_")[-1] == 'Nothing':
                Nothing += 1

            if str(results).split("_")[-1] == 'One_pair':
                One_pair += 1

            if str(results).split("_")[-1] == 'Three-of-a-kind':
                Three_of_a_Kind += 1

            if str(results).split("_")[-1] == 'Two-pair':
                Two_Pair += 1

            if str(results).split("_")[-1] == 'Full_house':
                Full_House += 1

            if str(results).split("_")[-1] == 'Four-of-a-kind':
                Four_of_a_Kind += 1

            if str(results).split("_")[-1] == 'Straight':
                Straight += 1

            if str(results).split("_")[-1] == 'Flush':
                Flush += 1

            if str(results).split("_")[-1] == 'Straight_Flush':
                Straight_Flush += 1
        except:
            error_num += 1
            continue

    frequency_hand_rank = {}
    frequency_hand_rank['High-Card'] = Nothing
    frequency_hand_rank['One-pair'] = One_pair
    frequency_hand_rank['Two-Pair'] = Two_Pair
    frequency_hand_rank['Three-of-a-Kind'] = Three_of_a_Kind
    frequency_hand_rank['Straight'] = Straight
    frequency_hand_rank['Flush'] = Flush
    frequency_hand_rank['Full-House'] = Full_House
    frequency_hand_rank['Straight-Flush'] = Straight_Flush
    frequency_hand_rank['Four-of-a-Kind'] = Four_of_a_Kind
    frequency_hand_rank['Straight-Flush'] = Straight_Flush

    poker_hand_ranks = frequency_hand_rank.keys()
    y_pos = np.arange(len(poker_hand_ranks))
    frequency = tuple(frequency_hand_rank.values())
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, frequency, color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(poker_hand_ranks)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Probability (%)")
    ax.set_title("Poker Hand Rank Probability (" + str(number_of_hands) + " hands)")
    plt.show()
    plt.savefig("{}.png".format(step))
    plt.close()

for step in ['one', 'two']:
    # step = one, 就是第一问，two就是第二题
    build_one(step=step)
    print("== " * 10)
