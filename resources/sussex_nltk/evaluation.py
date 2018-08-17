'''
.. codeauthor::
    Simon Wibberly
'''


def eval_sentiment(output_file):
    """
    evaluation routine for lang eng course sentiment analysis tasks
    
    should be in the format:
    
    [gold] [prediction]
    
    where tags are either P or N for positive and negative respectively
    
    """
    
    try:
        f = open(output_file)
    
    except IOError as ioe:
        print('error opening file', ioe)
    else :
        
        
        correct = 0.0
        incorrect = 0.0
        
        for line in f:
            split_line = line.split()
            if len(split_line) == 2:
                x, y = split_line
                
                if x == y:
                    correct += 1
                else:
                    incorrect += 1
                    
        try:
            accuracy = correct / (correct + incorrect)
        except ZeroDivisionError:
            accuracy = 0
        
        print("accuracy: %f" % ( accuracy ))
        
        return accuracy
        
    
    



