def generate_candidate_itemsets(level_k, level_frequent_itemsets, level_deleted_itemsets):
    """
    由频繁k-1项集生成候选k项集
    """

    # 获取频繁项集的数量
    n_frequent_itemsets = len(level_frequent_itemsets)
    candidate_itemsets = []

    # 遍历每一对频繁项集
    for i in range(n_frequent_itemsets):
        for j in range(i + 1, n_frequent_itemsets):
            itemset1 = list(level_frequent_itemsets[i])
            itemset2 = list(level_frequent_itemsets[j])
            itemset1.sort()
            itemset2.sort()

            # 如果前k-1项相同，则合并两个项集
            if itemset1[:level_k - 1] == itemset2[:level_k - 1]:
                combined_itemset = set(itemset1) | set(itemset2)

                # 检查是否有子集不在频繁项集中
                for deleted_itemset in level_deleted_itemsets:
                    if deleted_itemset.issubset(combined_itemset):
                        combined_itemset = None
                        break

                # 添加到候选项集
                if combined_itemset is not None:
                    candidate_itemsets.append(combined_itemset)

    return candidate_itemsets


def Apriori(dataset, min_sup):
    """
    Apriori算法
    """

    frequent_itemsets = []

    # 统计每个项的频数
    item_frequency = {}
    for transaction in dataset:
        for item in transaction:
            if item not in item_frequency:
                item_frequency[item] = 1
            else:
                item_frequency[item] += 1

    # 生成频繁1项集
    level_frequent_itemsets = []
    level_deleted_itemsets = []
    for item, count in item_frequency.items():
        if count >= min_sup:
            level_frequent_itemsets.append({item})
            frequent_itemsets.append([item])
        else:
            level_deleted_itemsets.append({item})

    k = 2

    while level_frequent_itemsets:
        # 由频繁k-1项集生成候选k项集
        candidate_itemsets = generate_candidate_itemsets(k - 1, level_frequent_itemsets, level_deleted_itemsets)

        # 统计候选k项集的频数
        itemset_count = {}
        for transaction in dataset:
            for itemset in candidate_itemsets:
                frozen_itemset = frozenset(itemset)
                if itemset.issubset(transaction):
                    if frozen_itemset not in itemset_count:
                        itemset_count[frozen_itemset] = 1
                    else:
                        itemset_count[frozen_itemset] += 1

        # 生成频繁k项集
        level_frequent_itemsets = []
        level_deleted_itemsets = []
        for itemset, count in itemset_count.items():
            if count >= min_sup:
                level_frequent_itemsets.append(set(itemset))
                frequent_itemsets.append(list(itemset))
            else:
                level_deleted_itemsets.append(set(itemset))

        k += 1

    return frequent_itemsets
