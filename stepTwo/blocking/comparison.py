""" Module with functionalities for comparison of attribute values as well as
    record pairs. The record pair comparison function will return a dictionary
    of the compared pairs to be used for classification.
"""

Q = 2  # Value length of q-grams for Jaccard and Dice comparison function


# =============================================================================
# First the basic functions to compare attribute values

def exact_comp(val1, val2):
    """Compare the two given attribute values exactly, return 1 if they are the
     same (but not both empty!) and 0 otherwise.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    elif (val1 != val2):
        return 0.0
    else:  # The values are the same
        return 1.0


# -----------------------------------------------------------------------------

def jaccard_comp(val1, val2):
    """Calculate the Jaccard similarity between the two given attribute values
     by extracting sets of sub-strings (q-grams) of length q.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement Jaccard similarity function here *********

    jacc_sim = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your Jaccard code *************************************

    assert jacc_sim >= 0.0 and jacc_sim <= 1.0

    return jacc_sim


# -----------------------------------------------------------------------------

def dice_comp(val1, val2):
    """Calculate the Dice coefficient similarity between the two given attribute
     values by extracting sets of sub-strings (q-grams) of length q.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement Dice similarity function here *********

    dice_sim = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your Dice code ****************************************

    assert dice_sim >= 0.0 and dice_sim <= 1.0

    return dice_sim


# -----------------------------------------------------------------------------

JARO_MARKER_CHAR = chr(1)  # Special character used in the Jaro, Winkler comp.


def jaro_comp(val1, val2):
    """Calculate the similarity between the two given attribute values based on
     the Jaro comparison function.

     As described in 'An Application of the Fellegi-Sunter Model of Record
     Linkage to the 1990 U.S. Decennial Census' by William E. Winkler and Yves
     Thibaudeau.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (val1 == '') or (val2 == ''):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    len1 = len(val1)  # Number of characters in val1
    len2 = len(val2)  # Number of characters in val2

    halflen = int(max(len1, len2) / 2) - 1

    assingment1 = ''  # Characters assigned in val1
    assingment2 = ''  # Characters assigned in val2

    workstr1 = val1  # Copy of original value1
    workstr2 = val2  # Copy of original value1

    common1 = 0  # Number of common characters
    common2 = 0  # Number of common characters

    for i in range(len1):  # Analyse the first string
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len2)
        index = workstr2.find(val1[i], start, end)
        if (index > -1):  # Found common character, count and mark it as assigned
            common1 += 1
            assingment1 = assingment1 + val1[i]
            workstr2 = workstr2[:index] + JARO_MARKER_CHAR + workstr2[index + 1:]

    for i in range(len2):  # Analyse the second string
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len1)
        index = workstr1.find(val2[i], start, end)
        if (index > -1):  # Found common character, count and mark it as assigned
            common2 += 1
            assingment2 = assingment2 + val2[i]
            workstr1 = workstr1[:index] + JARO_MARKER_CHAR + workstr1[index + 1:]

    if (common1 != common2):
        common1 = float(common1 + common2) / 2.0

    if (common1 == 0):  # No common characters within half length of strings
        return 0.0

    transposition = 0  # Calculate number of transpositions

    for i in range(len(assingment1)):
        if (assingment1[i] != assingment2[i]):
            transposition += 1
    transposition = transposition / 2.0

    common1 = float(common1)

    jaro_sim = 1. / 3. * (common1 / float(len1) + common1 / float(len2) + \
                          (common1 - transposition) / common1)

    assert (jaro_sim >= 0.0) and (jaro_sim <= 1.0), \
        'Similarity weight outside 0-1: %f' % (jaro_sim)

    return jaro_sim


# -----------------------------------------------------------------------------

def jaro_winkler_comp(val1, val2):
    """Calculate the similarity between the two given attribute values based on
     the Jaro-Winkler modifications.

     Applies the Winkler modification if the beginning of the two strings is
     the same.

     As described in 'An Application of the Fellegi-Sunter Model of Record
     Linkage to the 1990 U.S. Decennial Census' by William E. Winkler and Yves
     Thibaudeau.

     If the beginning of the two strings (up to first four characters) are the
     same, the similarity weight will be increased.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (val1 == '') or (val2 == ''):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # First calculate the basic Jaro similarity
    #
    jaro_sim = jaro_comp(val1, val2)
    if (jaro_sim == 0):
        return 0.0  # No common characters

    # ********* Implement Winkler similarity function here *********

    jw_sim = jaro_sim  # Replace with your code

    # Add your code here

    # ************ End of your Winkler code *************************************

    assert (jw_sim >= jaro_sim), 'Winkler modification is negative'
    assert (jw_sim >= 0.0) and (jw_sim <= 1.0), \
        'Similarity weight outside 0-1: %f' % (jw_sim)

    return jw_sim


# -----------------------------------------------------------------------------

def bag_dist_sim_comp(val1, val2):
    """Calculate the bag distance similarity between the two given attribute
     values.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement bag similarity function here *********
    # Extra task only

    bag_sim = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your bag distance code ********************************

    assert bag_sim >= 0.0 and bag_sim <= 1.0

    return bag_sim


# -----------------------------------------------------------------------------

def edit_dist_sim_comp(val1, val2):
    """Calculate the edit distance similarity between the two given attribute
     values.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement edit distance similarity here *********

    # Extra task only

    edit_sim = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your edit distance code *******************************

    assert edit_sim >= 0.0 and edit_sim <= 1.0

    return edit_sim


# -----------------------------------------------------------------------------

# Additional comparison functions for: (extra tasks for students to implement)
# - dates
# - ages
# - phone numbers
# - emails
# etc.

# =============================================================================
# Function to compare a block

def compareBlocks(blockA_dict, blockB_dict, recA_dict, recB_dict, \
                  attr_comp_list):
    """Build a similarity dictionary with pair of records from the two given
     block dictionaries. Candidate pairs are generated by pairing each record
     in a given block from data set A with all the records in the same block
     from dataset B.

     For each candidate pair a similarity vector is computed by comparing
     attribute values with the specified comparison method.

     Parameter Description:
       blockA_dict    : Dictionary of blocks from dataset A
       blockB_dict    : Dictionary of blocks from dataset B
       recA_dict      : Dictionary of records from dataset A
       recB_dict      : Dictionary of records from dataset B
       attr_comp_list : List of comparison methods for comparing individual
                        attribute values. This needs to be a list of tuples
                        where each tuple contains: (comparison function,
                        attribute number in record A, attribute number in
                        record B).

     This method returns a similarity vector with one similarity value per
     compared record pair.

     Example: sim_vec_dict = {(recA1,recB1) = [1.0,0.0,0.5, ...],
                              (recA1,recB5) = [0.9,0.4,1.0, ...],
                               ...
                             }
  """

    print('Compare %d blocks from dataset A with %d blocks from dataset B' % \
          (len(blockA_dict), len(blockB_dict)))

    sim_vec_dict = {}  # A dictionary where keys are record pairs and values
    # lists of similarity values

    # Iterate through each block in block dictionary from dataset A
    #
    for (block_bkv, rec_idA_list) in blockA_dict.items():

        # Check if the same blocking key occurs also for dataset B
        #
        if (block_bkv in blockB_dict):

            # If so get the record identifier list from dataset B
            #
            rec_idB_list = blockB_dict[block_bkv]

            # Compare each record in rec_id_listA with each record from rec_id_listB
            #
            for rec_idA in rec_idA_list:

                recA = recA_dict[rec_idA]  # Get the actual record A

                for rec_idB in rec_idB_list:
                    recB = recB_dict[rec_idB]  # Get the actual record B

                    # generate the similarity vector
                    #
                    sim_vec = compareRecord(recA, recB, attr_comp_list)

                    # Add the similarity vector of the compared pair to the similarity
                    # vector dictionary
                    #
                    sim_vec_dict[(rec_idA, rec_idB)] = sim_vec

    print('  Compared %d record pairs' % (len(sim_vec_dict)))
    print('')

    return sim_vec_dict


# -----------------------------------------------------------------------------

def compareRecord(recA, recB, attr_comp_list):
    """Generate the similarity vector for the given record pair by comparing
     attribute values according to the comparison function and attribute
     numbers in the given attribute comparison list.

     Parameter Description:
       recA           : List of first record values for comparison
       recB           : List of second record values for comparison
       attr_comp_list : List of comparison methods for comparing attributes,
                        this needs to be a list of tuples where each tuple
                        contains: (comparison function, attribute number in
                        record A, attribute number in record B).

     This method returns a similarity vector with one value for each compared
     attribute.
  """

    sim_vec = []

    # Calculate a similarity for each attribute to be compared
    #
    for (comp_funct, attr_numA, attr_numB) in attr_comp_list:

        if (attr_numA >= len(recA)):  # Check there is a value for this attribute
            valA = ''
        else:
            valA = recA[attr_numA]

        if (attr_numB >= len(recB)):
            valB = ''
        else:
            valB = recB[attr_numB]

        valA = valA.lower()
        valB = valB.lower()

        sim = comp_funct(valA, valB)
        sim_vec.append(sim)

    return sim_vec

# -----------------------------------------------------------------------------

# End of program.
