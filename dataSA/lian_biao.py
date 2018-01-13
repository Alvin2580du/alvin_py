# - * - coding:utf-8 - * -
"""定义节点类"""


class LNode:
    def __init__(self, x, next_=None):
        self.elem = x
        self.next = next_


"""
定义单链表类
"""


class LList:
    def __init__(self):
        self._head = None

    def length(self):
        count = 0  # 数目
        # 当前节点
        current = self._head
        while current != None:
            count += 1
            # 当前节点往后移
            current = current.next
        return count

    def is_empty(self):
        return self._head is None

    def prepend(self, elem):
        self._head = LNode(elem, self._head)

    """在链表头部添加元素"""

    def add(self, elem):
        node = LNode(elem)
        # 结点的next指针域指向头结点
        node.next = self._head
        # 头结点称为新的结点
        self._head = node

    def pop(self):
        if self._head is None:
            raise ValueError("in pop")
        e = self._head.elem
        self._head = self._head.next
        print("head is :%s" % self._head)
        return e

    # 后端操作
    def append(self, elem):
        # 当链表为空时
        if self._head is None:
            self._head = LNode(elem)
            return
        current = self._head
        while current.next is not None:
            current = current.next

        current.next = LNode(elem)

    def pop_last(self):
        if self._head is None:
            raise ValueError("in pop_last")
        p = self._head
        if p.next is None:
            e = p.elem
            self._head = None
            return e
        while p.next.next is not None:
            p = p.next
        e = p.next.elem
        p.next = None
        return e

    def find(self, pred):
        p = self._head
        while p is not None:
            if pred(p.elem):
                return p.elem
            p = p.next

    def printall(self):
        p = self._head
        while p is not None:
            if p.next is not None:
                print(', ')
            p = p.next


"""
定义循环单链表类
"""


class LCList:
    def __init__(self):
        self._head = None

    def is_empty(self):
        return self._head is None

    # 求链表长度
    def length(self):
        if self.is_empty():
            return 0
        count = 1  # 数目
        # 当前节点
        current = self._head
        # 当前节点的下一个节点不是头结点则继续增加
        while current.next != self._head:
            count += 1
            # 当前节点往后移
            current = current.next
        return count

    # add(elem) 链表头部添加元素
    def add(self, elem):
        node = LNode(elem)
        if self.is_empty():
            # 空链表
            self.__head = node
            node.next = node
        else:
            # 非空链表添加
            current = self.__head
            # 查找最后一个节点
            while current.next != self.__head:
                current = current.next
            # 新节点的下一个节点为旧链表的头结点
            node.next = self.__head
            # 新链表的头结点为新节点
            self.__head = node
            # 最后节点的下一个节点指向新节点
            current.next = node

    def prepend(self, elem):
        p = LNode(elem)
        # 如果为空
        if self._head is None:
            p.next = p
            self._head = p
        else:
            p.next = self._head.next
            self._head.next = p

    def append(self, elem):
        self.prepend(elem)
        self._head = self._head.next

    def pop(self):
        if self._head is None:
            raise ValueError("in pop of CLList")
        p = self._head.next
        if self._head is p:
            self._head = None
        else:
            self._head.next = p.next
        return p.elem

    # search(elem) 查找节点是否存在
    def search(self, elem):
        # 当前节点
        if self.is_empty():
            # 空链表直接返回False
            return False
        current = self.__head
        while current.next != self.__head:
            if current.elem == elem:
                # 找到了
                return True
            else:
                current = current.next
        # 判断最后一个元素
        if current.elem == elem:
            return True
        return False

    def printall(self):
        if self.is_empty():
            return
        p = self._head.next
        while True:
            print (p.elem)
            if p is self._head:
                break
            p = p.next


"""定义双链表结点类，在LNode的基础上派生类"""


class DLNode(LNode):
    def __init__(self, elem, prev=None, next_=None):
        LNode.__init__(self, elem, next_)
        self.prev = prev


"""
定义双链表类
"""


class Double_LList(LList):
    def __init__(self):
        LList.__init__(self)

    def is_empty(self):
        return self._head is None
        # length() 链表长度

    def length(self):
        count = 0  # 数目
        # 当前节点
        current = self._head
        while current != None:
            count += 1
            # 当前节点往后移
            current = current.next
        return count

    # add(elem) 链表头部添加元素
    def add(self, elem):
        node = DLNode(elem)
        # 新节点的下一个节点为旧链表的头结点
        node.next = self.__head
        # 新链表的头结点为新节点
        self.__head = node
        # 下一个节点的上一个节点指向新增的节点
        # 相当于是第一个结点指向新添加结点
        node.next.prev = node

    def prepend(self, elem):
        p = DLNode(elem, None, self._head)
        if self._head is None:
            self._head = p
        else:
            p.prev.prev = p
        self._head = p

    def append(self, elem):
        p = DLNode(elem, self._head, None)
        if self._head is None:
            self._head = p
        else:
            p.prev.next = p
        self._head = p

    def pop(self):
        if self._head is None:
            raise ValueError("in pop_last of DDList")
        e = self._head.elem
        self._head = self._head.next
        if self._head is not None:
            self._head.prev = None
        return e

    def pop_last(self):
        if self._head is None:
            raise ValueError("in pop_last of DLList")

        e = self._head.elem
        self._head = self._head.prev
        if self._head is None:
            self._head = None
        else:
            self._head.next = None
        return e


