from collections import OrderedDict, Counter


class FPTree(object):
    def __init__(self, root=None):
        '''
        :param root: `FPTreeNode` instance which is the root element
        '''
        self._headers = OrderedDict()
        self._latest_ptr = OrderedDict()
        if root is None:
            self.root = FPTreeNode()
        else:
            self.root = root

    def _update_pointers(self, node):
        ''' update the linked-list and header pointers for a given nodes of
        specific items
        :param node: `FPTreeNode` instance to update the header table and other
            node links.
        '''
        item = node.item
        # add the item in the header if not already present.
        if item not in self._headers:
            self._headers[item] = node
            self._latest_ptr[item] = node
        else:
            # if already present in the header, that means we should change the
            # entry in the _latest_ptr, and make current latest point to `node`.
            latest_item = self._latest_ptr[item]
            latest_item.neighbor = node
            # make the current `node` latest
            self._latest_ptr[item] = node

    def insert_transaction(self, transaction):
        '''
        :param transaction: It is a list of items to be inserted into
        '''
        current = self.root
        for item in transaction:
            if current.has_child(item):
                current = current.get_child(item)
                current.count += 1
            else:
                # create a new node
                new_node = current.add_child(item)
                new_node.parent = current
                current = new_node
                self._update_pointers(current)

    def prefix_path(self, item):
        ''' return the prefix paths containing a particular item '''
        link_node = self._headers[item]
        paths = []
        while link_node:
            path = []
            path_node = link_node.parent
            while not path_node.root:
                path.append(path_node)
                path_node = path_node.parent
            paths.append(path)
            link_node = link_node.neighbor
        return paths

    @property
    def headers(self):
        ''' iterator which returns the header nodes in reverse manner '''
        for node in reversed(self._headers.values()):
            yield node

    def __iter__(self):
        '''
        a breadth first search iterator for the tree
        '''
        to_visit = [self.root]
        for node in to_visit:
            to_visit.extend(node.children.values())
            yield node

    def __str__(self):
        str_ = ''
        for node in self:
            res = [str(child) for child in node.children.values()]
            if len(res) > 0:
                str_ = str_ + "\n%s -> (%s)" % (node, res)
        return str_


class FPTreeNode(object):
    def __init__(self, item=None):
        # children of the nodes in the tree
        self.children = OrderedDict()
        # points to the next node in the tree corresponding to the same item
        self.neighbor = None
        self.parent = None

        if item is None:
            self.count = 0
            self.item = None
            self._root = True
        else:
            self.count = 1
            self.item = item
            self._root = False

    @property
    def root(self):
        return self._root

    def has_child(self, item):
        '''
        return if the current node has a child with given item
        '''
        return item in self.children

    def get_child(self, item):
        '''
        get the child with the given item, raise a `KeyError` otherwise
        '''
        return self.children[item]

    def add_child(self, item):
        '''
        create a child of the current node with the following item, raises a
        `ValueError` if a child exists with the given item.
        '''
        if item not in self.children:
            child = FPTreeNode(item)
            self.children[item] = child
            return child
        else:
            raise ValueError

    def __str__(self):
        return "(%s: %s)" % (self.item, self.count)


def build_fp_tree(transactions, min_support):
    ''' a function to build a FP tree given a set of transactions,
    the itemsets will be frequency analyzed and the transactions will be
    sorted before building the FP tree '''
    c = Counter()
    # get the number of occurences for each of the items
    for transaction in transactions:
        c.update(transaction)
    # get all items in sorted order and greater than or equal
    # minimum support.
    item_frequencies = OrderedDict((item, support) for item, support in c.most_common() if support >= min_support)
    # function for sorting the items in the transaction
    sort_fn = lambda item: list(item_frequencies.keys()).index(item)
    # function for filtering the items in the transaction
    filter_fn = lambda item: item in item_frequencies
    # sort all the transactions
    sorted_transactions = (sorted(filter(filter_fn, transaction), key=sort_fn) for transaction in transactions)
    # build the tree
    tree = FPTree()
    for transaction in sorted_transactions:
        tree.insert_transaction(transaction)

    return tree


def readFile(filename):
    """ 
    Function read the file 
    return a list of sets which contains the information of the transaction
    """
    originalList = list()
    file = open(filename, 'rU')
    c = 0
    limit = 1000
    for line in file:
        c = c + 1
        line = line.strip().rstrip(',')
        record = set(line.split(', '))
        originalList.append(record)
        if c > limit:
            break
    return originalList, c

if __name__ == '__main__':
    transactions = readFile('adult.data.txt')
    print(len(transactions))
    fp_tree = build_fp_tree(transactions, 3)
    print(fp_tree)
