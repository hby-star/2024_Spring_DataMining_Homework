from collections import OrderedDict


class TreeNode:
    """
    FP树节点
    """

    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 节点名称
        self.count = numOccur  # 节点出现的次数
        self.nodeLink = None  # 指向下一个相似节点
        self.parent = parentNode  # 指向父节点
        self.children = {}  # 指向子节点，子节点元素名称为键，指向子节点指针为值

    def inc(self, numOccur):
        self.count += numOccur


def createInitSet(dataSet):
    """
    初始化数据集
    """
    retDict = OrderedDict()
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def createTree(dataSet, minSup):
    """
    生成FP-tree
    """

    # 第一次遍历数据集，统计每个元素出现的次数
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # 删除不满足最小支持度的元素
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None

    # 添加一个数据项，用于存放指向相似元素项的指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = TreeNode('Null', 1, None)

    # 第二次遍历数据集，构建FP-tree
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    """
    更新FP树
    """

    if items[0] in inTree.children:
        # 如果该元素在树的子节点中，计数增加
        inTree.children[items[0]].inc(count)
    else:
        # 如果该元素不在树的子节点中，创建一个新的子节点
        inTree.children[items[0]] = TreeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            # 如果该元素在项头表中的链表为空，直接指向该节点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 如果该元素在项头表中的链表不为空，指向链表的末尾
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 如果仍有未分配完的元素，递归调用updateTree
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    """
    更新头指针表
    """

    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    """
    迭代上溯整棵树
    """

    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    """
    寻找前缀路径
    """

    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(headerTable, minSup, preFix, freqItemList):
    """
    挖掘频繁项集
    """

    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)

        freqItemList.append(list(newFreqSet))

        condPathBases = findPrefixPath(headerTable[basePat][1])
        myCondTree, myHead = createTree(condPathBases, minSup)
        if myHead is not None:
            mineTree(myHead, minSup, newFreqSet, freqItemList)


def FPGrowth(dataset: list, min_sup: int):
    """
    FPGrowth算法
    """

    freqItems = []
    FPTree, headerTable = createTree(createInitSet(dataset), min_sup)
    mineTree(headerTable, min_sup, set([]), freqItems)
    return freqItems
