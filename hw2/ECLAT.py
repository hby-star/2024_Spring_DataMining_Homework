def eclat_recur(prefix, items, min_support, freq_items):
    """
    eclat的递归实现
    """

    while items:
        key, item = items.pop()
        key_support = len(item)

        if key_support >= min_support:
            # 添加频繁项集
            freq_items.append(prefix + list(sorted([key])))
            new_itemsets = []
            for other_key, other_item in items:
                # 求key与其他项集的交集
                new_item = item & other_item
                # 保存不小于最小支持度的项集
                if len(new_item) >= min_support:
                    new_itemsets.append((other_key, new_item))
            # 对由key与其他项集交集得到的项集递归调用
            eclat_recur([key], sorted(new_itemsets, key=lambda item: len(item[1]), reverse=True), min_support,
                        freq_items)


def ECLAT(dataset, min_support):
    """
    ECLAT算法
    """

    # 转换为垂直数据格式
    data = {}
    trans_num = 0
    for trans in dataset:
        trans_num += 1
        for item in trans:
            if item not in data:
                data[item] = set()
            data[item].add(trans_num)

    freq_items = []
    eclat_recur([], sorted(data.items(), key=lambda item: len(item[1]), reverse=True), min_support,
                freq_items)
    return freq_items
