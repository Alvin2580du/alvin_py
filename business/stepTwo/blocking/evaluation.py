""" Module with functionalities to evaluate the results of a record linkage
    excercise, both with reagrd to linkage quality as well as complexity.
"""


# =============================================================================

def confusion_matrix(class_match_set, class_nonmatch_set, true_match_set,
                     all_comparisons):
    """Compute the confusion (error) matrix which has the following form:

     +-----------------+-----------------------+----------------------+
     |                 |  Predicted Matches    | Predicted NonMatches |
     +=================+=======================+======================+
     | True  Matches   | True Positives (TP)   | False Negatives (FN) |
     +-----------------+-----------------------+----------------------+
     | True NonMatches | False Positives (FP)  | True Negatives (TN)  |
     +-----------------+-----------------------+----------------------+

     The four values calculated in the confusion matrix (TP, FP, TN, and FN)
     are then the basis of linkag equality measures such as precision and
     recall.

     Parameter Description:
       class_match_set    : Set of classified matches (record identifier
                            pairs)
       class_nonmatch_set : Set of classified non-matches (record identifier
                            pairs)
       true_match_set     : Set of true matches (record identifier pairs)
       all_comparisons    : The total number of comparisons between all record
                            pairs

     This function returns a list with four values representing TP, FP, FN,
     and TN.
  """

    print('Calculating confusion matrix using %d classified matches, %d ' % \
          (len(class_match_set), len(class_nonmatch_set)) + 'classified ' + \
          'non-matches, and %d true matches' % (len(true_match_set)))

    num_tp = 0  # number of true positives
    num_fp = 0  # number of false positives
    num_tn = 0  # number of true negatives
    num_fn = 0  # number of false negatives

    # Iterate through the classified matches to check if they are true matches or
    # not
    #
    for rec_id_tuple in class_match_set:
        if (rec_id_tuple in true_match_set):
            num_tp += 1
        else:
            num_fp += 1

    # Iterate through the classified non-matches to check of they are true
    # non-matches or not
    #
    for rec_id_tuple in class_nonmatch_set:

        # Check a record tuple is only counted once
        #
        assert rec_id_tuple not in class_match_set, rec_id_tuple

        if (rec_id_tuple in true_match_set):
            num_fn += 1
        else:
            num_tn += 1

    # Finally count all missed true matches to the false negatives
    #
    for rec_id_tuple in true_match_set:
        if ((rec_id_tuple not in class_match_set) and \
                    (rec_id_tuple not in class_nonmatch_set)):
            num_fn += 1

    num_tn = all_comparisons - num_tp - num_fp - num_fn

    print('  TP=%s, FP=%d, FN=%d, TN=%d' % (num_tp, num_fp, num_fn, num_tn))
    print('')

    return [num_tp, num_fp, num_fn, num_tn]


# =============================================================================
# Different linkage quality measures

def accuracy(confusion_matrix):
    """Compute accuracy using the given confusion matrix.

     Accuracy is calculated as (TP + TN) / (TP + FP + FN + TN).

     Parameter Description:
       confusion_matrix : The matrix with TP, FP, FN, TN values.

     The method returns a float value.
  """

    num_tp = confusion_matrix[0]
    num_fp = confusion_matrix[1]
    num_fn = confusion_matrix[2]
    num_tn = confusion_matrix[3]

    accuracy = float(num_tp + num_tn) / (num_tp + num_fp + num_fn + num_tn)

    return accuracy


# -----------------------------------------------------------------------------

def precision(confusion_matrix):
    """Compute precision using the given confusion matrix.

     Precision is calculated as TP / (TP + FP).

     Parameter Description:
       confusion_matrix : The matrix with TP, FP, FN, TN values.

     The method returns a float value.
  """

    # ************************ Implement precision here *************************

    precision = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your precision code ***********************************

    return precision


# -----------------------------------------------------------------------------

def recall(confusion_matrix):
    """Compute recall using the given confusion matrix.

     Recall is calculated as TP / (TP + FN).

      Parameter Description:
        confusion_matrix : The matrix with TP, FP, FN, TN values.

      The method returns a float value.
  """

    # ************************ Implement precision here *************************

    recall = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your recall code **************************************

    return recall


# -----------------------------------------------------------------------------

def fmeasure(confusion_matrix):
    """Compute the f-measure of the linkage.

     The f-measure is calculated as:

              2 * (precision * recall) / (precision + recall).

     Parameter Description:
       confusion_matrix : The matrix with TP, FP, FN, TN values.

     The method returns a float value.
  """
    # ************************ Implement precision here *************************

    f_measure = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your f-measure code ***********************************

    return f_measure


# =============================================================================
# Different linkage complexity measures

def reduction_ratio(num_comparisons, all_comparisons):
    """Compute the reduction ratio using the given confusion matrix.

     Reduction ratio is calculated as 1 - num_comparison / (TP + FP + FN+ TN).

     Parameter Description:
       num_comparisons : The number of candidate record pairs
       all_comparisons : The total number of comparisons between all record
                         pairs

     The method returns a float value.
  """

    if (num_comparisons == 0):
        return 1.0

    rr = 1.0 - float(num_comparisons) / all_comparisons

    return rr


# -----------------------------------------------------------------------------

def pairs_completeness(cand_rec_id_pair_list, true_match_set):
    """Pairs completeness measures the effectiveness of a blocking technique in
     the record linkage process.

     Pairs completeness is calculated as the number of true matches included in
     the candidate record pairs divided by the number of all true matches.

     Parameter Description:
       cand_rec_id_pair_list : Dictionary of candidate record pairs generated
                               by a blocking technique
       true_match_set        : Set of true matches (record identifier pairs)

     The method returns a float value.
  """

    # ************************ Implement precision here *************************

    pc = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your pairs completeness code **************************

    return pc


# -----------------------------------------------------------------------------

def pairs_quality(cand_rec_id_pair_list, true_match_set):
    """Pairs quality measures the efficiency of a blocking technique.

     Pairs quality is calculated as the number of true matches included in the
     candidate record pairs divided by the number of candidate record pairs
     generated by blocking.

     Parameter Description:
       cand_rec_id_pair_list : Dictionary of candidate record pairs generated
                               by a blocking technique
       true_match_set        : Set of true matches (record identifier pairs)

     The method returns a float value.
  """

    # ************************ Implement precision here *************************

    pq = 0.0  # Replace with your code

    # Add your code here

    # ************ End of your pairs quality code *******************************

    return pq

# -----------------------------------------------------------------------------

# End of program.
