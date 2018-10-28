""" Module with functionalities for classifying a dictionary of record pairs
    and their similarities.

    Each function in this module returns two sets, one with record pairs
    classified as matches and the other with record pairs classified as
    non-matches.
"""

# =============================================================================

def exactClassify(sim_vec_dict):
  """Method to classify the given similarity vector dictionary assuming only
     exact matches (having all similarities of 1.0) are matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.

     The classification is based on the exact matching of attribute values,
     that is the similarity vector for a given record pair must contain 1.0
     for all attribute values.

     Example:
       (recA1, recB1) = [1.0, 1.0, 1.0, 1.0] => match
       (recA2, recB5) = [0.0, 1.0, 0.0, 1.0] = non-match
  """

  print('Exact classification of %d record pairs' % (len(sim_vec_dict)))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    sim_sum = sum(sim_vec)  # Sum all attribute similarities

    if sim_sum == len(sim_vec):  # All similarities were 1.0
      class_match_set.add(rec_id_tuple)
    else:
      class_nonmatch_set.add(rec_id_tuple)

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def thresholdClassify(sim_vec_dict, sim_thres):
  """Method to classify the given similarity vector dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where record pairs
     with an average similarity of at least this threshold are classified as
     matches and all others as non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       sim_thres    : The classification similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  print('Similarity threshold based classification of %d record pairs' % \
        (len(sim_vec_dict)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    # ********* Implement threshold based classification **********************

    pass  # Add your code here 

    # ************ End of your code *******************************************

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def minThresholdClassify(sim_vec_dict, sim_thres):
  """Method to classify the given similarity vector dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where record pairs
     that have all their similarities (of all attributes compared) with at
     least this threshold are classified as matches and all others as
     non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       sim_thres    : The classification minimum similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  print('Minimum similarity threshold based classification of ' + \
        '%d record pairs' % (len(sim_vec_dict)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    # ********* Implement minimum threshold classification ********************

    pass  # Add your code here 

    # ************ End of your code *******************************************

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def weightedSimilarityClassify(sim_vec_dict, weight_vec, sim_thres):
  """Method to classify the given similarity vector dictionary with regard to
     a given weight vector and a given similarity threshold (in the range 0.0
     to 1.0), where an overall similarity is calculated based on the weights
     for each attribute, and where record pairs with the similarity of at least
     the given threshold are classified as matches and all others as
     non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       weight_vec   : A vector with weights, one weight for each attribute.
       sim_thres    : The classification similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  # Check weights are available for all attributes
  #
  first_sim_vec = list(sim_vec_dict.values())[0]
  assert len(weight_vec) == len(first_sim_vec), len(weight_vec)

  print('Weighted similarity based classification of %d record pairs' % \
        (len(sim_vec_dict)))
  print('  Weight vector: %s'   % (str(weight_vec)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  weight_sum = sum(weight_vec)  # Sum of all attribute weights

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    # ******* Implement weighted similarity classification ********************

    pass  # Add your code here 

    # ************ End of your code *******************************************

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def supervisedMLClassify(sim_vec_dict, true_match_set):
  """A classifier method based on a supervised machine learning technique
     (decision tree) which learns from the given similarity vectors and the
     true match status set provided.

     The approach works as follows:
     1) Create the matrix of features (similarity vectors) and class labels 
        (true matches and true non-matches)
     2) Generate 3 or 5 or 7 etc. decision tree classifiers as follows:
        2a) Sample 2/3 of all training examples -> training set
            The remaining training examples ->     test set
        2b) Train the decision tree classifier on the training set
        2c) Test the accuracy of the classifier on the test set
     3) For each record pair and its similarity vector, apply the 3 or 5
        trained classifiers, get the majority class (match or non-match) as
        its final class

     Parameter Description:
       sim_vec_dict  : Dictionary of record pairs with their identifiers as
                       as keys and their corresponding similarity vectors as
                       values.
       true_mach_set : Set of true matches (record identifier pairs)
  """

  num_folds = 3  # Number of classifiers to create

  class_match_set =    set()
  class_nonmatch_set = set()

  try:
    import numpy
    import sklearn.tree
  except:
    print('Either the "numpy" or "sklearn" modules is not installed! Aborting.')
    print('')

    return set(), set()  # Return two empty sets so program continues

  import random

  print('Supervised decision tree classification of %d record pairs' % \
        (len(sim_vec_dict)))

  # Generate the training data sets (similarity vectors plus class labels
  # (match or non-match)
  #
  num_train_rec = len(sim_vec_dict)
  num_features =  len(list(sim_vec_dict.values())[0])

  print('  Number of training records and features: %d / %d' % \
        (num_train_rec, num_features))

  all_train_data =  numpy.zeros([num_train_rec, num_features])
  all_train_class = numpy.zeros(num_train_rec)

  rec_pair_id_list = []

  num_pos = 0
  num_neg = 0

  i = 0
  for (rec_id1,rec_id2) in sim_vec_dict:
    rec_pair_id_list.append((rec_id1,rec_id2))
    sim_vec = sim_vec_dict[(rec_id1,rec_id2)]

    all_train_data[:][i] = sim_vec

    if (rec_id1,rec_id2) in true_match_set:
      all_train_class[i] = 1.0
      num_pos += 1
    else:
      all_train_class[i] = 0.0
      num_neg += 1
    i += 1

  num_all = num_pos + num_neg  # All training examples

  num_train_select = int(2./3 * num_all)  # Select 2/3 for training
  num_test_select =  num_all - num_train_select

  print('  Number of positive and negative training records: %d / %d' % \
        (num_pos, num_neg))
  print('')

  class_list = []  # List of the generated classifiers

  for c in range(num_folds):

    train_index_list = random.sample(xrange(num_all), num_train_select)

    train_data =  numpy.zeros([num_train_select, num_features])
    train_class = numpy.zeros(num_train_select)
    test_data =   numpy.zeros([num_test_select, num_features])
    test_class =  numpy.zeros(num_test_select)

    # Copy similarities and class labels
    #
    train_ind = 0
    test_ind =  0

    for i in range(num_all):

      if (i in train_index_list):
        train_data[:][train_ind] = all_train_data[:][i]
        train_class[train_ind] =   all_train_class[i]
        train_ind += 1
      else:
        test_data[:][test_ind] = all_train_data[:][i]
        test_class[test_ind] =   all_train_class[i]
        test_ind += 1

    # Now build and train the classifier
    #
    decision_tree = sklearn.tree.DecisionTreeClassifier()
    decision_tree.fit(train_data, train_class)

    # Now use the trained classifier on the testing data to see how accurate
    # it is
    #
    class_predict = decision_tree.predict(test_data)

    num_corr =  0
    num_wrong = 0

    for i in range(len(class_predict)):
      if (class_predict[i] == test_class[i]):
        num_corr += 1
      else:
        num_wrong += 1

    print('  Classifier %d gets %d correct and %d wrong' % \
          (c, num_corr, num_wrong))

    class_list.append(decision_tree)

  # Now use the trained classifiers to classify all record pairs
  #
  num_match_class_list = [0]*num_all  # Count how often a record pair is
                                      # classified as a match

  for decision_tree in class_list:

    class_predict = decision_tree.predict(all_train_data)  # Classify all pairs

    for i in range(num_all):
      num_match_class_list[i] += class_predict[i]

      assert num_match_class_list[i] <= num_folds, num_match_class_list[i]

  for i in range(num_all):
    rec_id_pair = rec_pair_id_list[i]

    # More '1' (match) classifications than '0' (non-match ones)
    #
    if (float(num_match_class_list[i]) / num_folds > 0.5):
      class_match_set.add(rec_id_pair)
    else:
      class_nonmatch_set.add(rec_id_pair)

  print('')

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

# End of program.
