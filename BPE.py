import collections

class BPETokenizerWithID:
    def __init__(self):
        # token_to_id: 词表，存每个 token 对应的 ID
        # id_to_token: 反向词表，用于解码
        self.token_to_id = {}
        self.id_to_token = {}
        
        # merges: 记录合并规则 {(id1, id2): new_id}
        self.merges = {}
        
        # 特殊 token
        self.UNK_TOKEN = "<unk>" # 未知字符
        self.END_TOKEN = "</w>"  # 单词结束符

    def build_base_vocab(self, corpus_words):
        """
        第一步：给语料中出现的所有单个字符分配 ID
        """
        # 1. 收集所有唯一的字符
        unique_chars = set()
        for word in corpus_words:
            unique_chars.update(word)
        
        # 2. 排序，保证 ID 分配是确定的
        sorted_chars = sorted(list(unique_chars))
        
        # 3. 分配基础 ID (从 0 开始)
        # 先加特殊 token
        self.token_to_id[self.UNK_TOKEN] = 0
        self.token_to_id[self.END_TOKEN] = 1
        
        idx = 2
        for char in sorted_chars:
            if char not in self.token_to_id: # 避免覆盖特殊 token
                self.token_to_id[char] = idx
                idx += 1
                
        # 生成反向表
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        print(f"基础词表构建完成，基础字符数: {len(self.token_to_id)}")
        print(f"示例: 'a'->{self.token_to_id.get('a')}, 'b'->{self.token_to_id.get('b')}")

    def get_stats(self, ids_list):
        """
        统计 ID 对的频率
        ids_list: [[10, 12, 1], [10, 15, 1], ...]  (即单词的 ID 序列列表)
        """
        pairs = collections.defaultdict(int)
        for ids in ids_list:
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                pairs[pair] += 1
        return pairs

    def merge_ids(self, ids_list, pair, new_id):
        """
        在所有 ID 序列中，将 (id1, id2) 替换为 new_id
        """
        new_ids_list = []
        p1, p2 = pair
        
        for ids in ids_list:
            new_ids = []
            i = 0
            while i < len(ids):
                # 如果发现当前位置和下一个位置正好匹配我们要合并的对
                if i < len(ids) - 1 and ids[i] == p1 and ids[i+1] == p2:
                    new_ids.append(new_id) # 放入新的 ID
                    i += 2 # 跳过这两个旧 ID
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_list.append(new_ids)
            
        return new_ids_list

    def train(self, corpus, num_merges=5):
        print(f"\n--- 开始训练 (目标合并: {num_merges} 次) ---")
        
        # 1. 预处理：把句子拆成单词，并加上结束符
        words = [list(w) + [self.END_TOKEN] for w in corpus.split()]
        
        # 2. 构建基础词表 (给字符分配 ID)
        self.build_base_vocab(words)
        
        # 3. 将单词转换为初始 ID 列表
        # 例如: [['h', 'u', 'g', '</w>'], ...] -> [[10, 25, 18, 1], ...]
        corpus_ids = []
        for word in words:
            word_ids = [self.token_to_id.get(c, 0) for c in word]
            corpus_ids.append(word_ids)

        # 4. 循环合并
        for i in range(num_merges):
            # 统计频率
            pairs = self.get_stats(corpus_ids)
            if not pairs:
                break
                
            # 找到频率最高的 ID 对
            best_pair = max(pairs, key=pairs.get) # 例如 (10, 25)
            best_freq = pairs[best_pair]
            
            # --- 关键点：分配新 ID ---
            new_id = len(self.token_to_id) # 新 ID 就是当前词表长度 (即往后追加)
            
            # 生成新 token 的字符串表示 (仅用于显示)
            token_str = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            
            # 记录规则
            self.merges[best_pair] = new_id
            self.token_to_id[token_str] = new_id
            self.id_to_token[new_id] = token_str
            
            # 执行替换
            corpus_ids = self.merge_ids(corpus_ids, best_pair, new_id)
            
            print(f"Round {i+1}: 合并 ID {best_pair} -> 新 ID {new_id} (Token: '{token_str}'), 频次: {best_freq}")

        print("--- 训练结束 ---")

    def encode(self, text):
        """
        推理：将文本转换为 ID 列表
        """
        # 1. 初始拆分
        words = [list(w) + [self.END_TOKEN] for w in text.split()]
        
        encoded_ids = []
        
        for word in words:
            # 2. 转为基础 ID
            ids = [self.token_to_id.get(c, self.token_to_id[self.UNK_TOKEN]) for c in word]
            
            # 3. 按照优先级应用合并规则
            while len(ids) >= 2:
                stats = self.get_stats([ids]) # 检查当前单词里有哪些对
                
                # 找出这些对里，哪一个是我们可以合并的 (且是 ID 最小的优先，模拟训练顺序)
                pair_to_merge = None
                min_new_id = float('inf')
                
                for pair in stats:
                    if pair in self.merges:
                        # 找到对应的新 ID
                        target_id = self.merges[pair]
                        # 我们希望优先合并“最早训练出来”的规则，因为新 ID 是递增的
                        # 所以 ID 越小，说明越早被训练出来
                        if target_id < min_new_id:
                            min_new_id = target_id
                            pair_to_merge = pair
                
                if pair_to_merge:
                    # 执行一次合并
                    ids = self.merge_ids([ids], pair_to_merge, min_new_id)[0]
                else:
                    break # 没得合了
            
            encoded_ids.extend(ids)
            
        return encoded_ids

    def decode(self, ids):
        """
        解码：ID 列表 -> 文本
        """
        tokens = [self.id_to_token.get(idx, "") for idx in ids]
        # 拼起来，并处理 </w> (替换为空格)
        text = "".join(tokens).replace(self.END_TOKEN, " ")
        return text.strip()

# =================测试代码=================
if __name__ == '__main__':
    # 语料：hug (抱) 和 bug (虫) 和 huge (巨大的)
    corpus = "hug " * 10 + "bug " * 5 + "huge " * 5
    
    tokenizer = BPETokenizerWithID()
    tokenizer.train(corpus, num_merges=3)
    
    print("\n[词表查看]")
    # 打印 ID > 10 的部分 (即合并出来的新 token)
    for k, v in tokenizer.token_to_id.items():
        if v > 8: # 过滤掉部分基础字符，只看新合并的
            print(f"Token: {k:8} | ID: {v}")

    print("\n[推理测试]")
    # 这是一个训练集里没有的词，但它包含训练过的子词 'hug'
    test_text = "thug" 
    ids = tokenizer.encode(test_text)
    print(f"原文: {test_text}")
    print(f"编码(ID): {ids}")
    print(f"解码(Str): {tokenizer.decode(ids)}")
