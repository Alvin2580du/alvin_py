""" Module with functionality for saving a CSV file with the linkage results.
"""

# =============================================================================
# Import necessary modules

import os


# -----------------------------------------------------------------------------

def save_linkage_set(file_name, class_match_set):
    """Write the given set of matches (record pair identifiers) into a CSV file
     with one pair of record identifiers per line)

     Parameter Description:
       file_name       : Name of the data file to be write into (a CSV file)
       class_match_set : The set of classified matches (pairs of record
                         identifiers) 
  """

    out_f = open(file_name, 'w')  # Open a CSV file for writing

    print('Write linkage results to file: ' + file_name)

    for (rec_id1, rec_id2) in sorted(class_match_set):  # Sort for nicer output

        out_line = '%s, %s' % (rec_id1, rec_id2)

        out_f.write(out_line + os.linesep)

    out_f.close()

    print('  Wrote %d linked record pairs' % (len(class_match_set)))
    print('')

# -----------------------------------------------------------------------------

# End of program.
