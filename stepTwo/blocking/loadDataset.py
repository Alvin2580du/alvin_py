""" Module with functionalities for reading data from a file and return a
    dictionary with record identifiers as keys and a list of attribute values.

    Also provides a function to load a truth data set of record pairs that are
    matches.
"""

# =============================================================================
# Import necessary modules

import csv
import gzip


# -----------------------------------------------------------------------------

def load_data_set(file_name, rec_id_col, use_attr_list, header_line):
    """Load the data set and store in memory as a dictionary with record
     identifiers as keys.

     Parameter Description:
       file_name      : Name of the data file to be read (CSV or CSV.GZ file)
       rec_id_col     : Record identifier column of the data file
       use_attr_list  : List of attributes to extract from the file
       header_line    : Availability of the header line (True of False)
  """

    # Open a CSV for Gzipped (compressed) CSV.GZ file
    #
    if (file_name.endswith('gz')):
        in_f = gzip.open(file_name, 'rt')
    else:
        in_f = open(file_name)

    csv_reader = csv.reader(in_f)

    print('Load data set from file: ' + file_name)

    if (header_line == True):
        header_list = next(csv_reader)
        print('  Header line: ' + str(header_list))

    print('  Record identifier attribute: ' + str(header_list[rec_id_col]))
    print('  Attributes to use:')
    for attr_num in use_attr_list:
        print('    ' + header_list[attr_num])

    rec_num = 0
    rec_dict = {}

    # Iterate through the record in the file
    #
    for rec_list in csv_reader:
        rec_num += 1

        # Get the record identifier
        #
        rec_id = rec_list[rec_id_col].strip().lower()

        rec_val_list = []  # One value list per record

        for attr_id in range(len(rec_list)):
            if attr_id in use_attr_list:
                rec_val_list.append(rec_list[attr_id].strip().lower())
            else:
                rec_val_list.append('')

        rec_dict[rec_id] = rec_val_list

    in_f.close()

    if (len(rec_dict) < rec_num):
        print('  *** Warning, data set contains %d duplicates ***' % \
              (rec_num - len(rec_dict)))
        print('       %d unique records' % (len(rec_dict)))

    print('')

    # Return the generated dictionary of records
    #
    return rec_dict


# -----------------------------------------------------------------------------

def load_truth_data(file_name):
    """Load a truth data file where each line contains two record identifiers
     where the corresponding record pair is a true match.

     Returns a set where the elements are pairs (tuples) of these record
     identifier pairs.
  """

    if (file_name.endswith('gz')):
        in_f = gzip.open(file_name, 'rt')
    else:
        in_f = open(file_name)

    csv_reader = csv.reader(in_f)

    print('Load truth data from file: ' + file_name)

    truth_data_set = set()

    for rec_list in csv_reader:
        assert len(rec_list) == 2, rec_list  # Make sure only two identifiers

        rec_id1 = rec_list[0].lower()
        rec_id2 = rec_list[1].lower()

        truth_data_set.add((rec_id1, rec_id2))

    in_f.close()

    print('  Loaded %d true matching record pairs' % (len(truth_data_set)))
    print('')

    return truth_data_set

# -----------------------------------------------------------------------------

# End of program.
