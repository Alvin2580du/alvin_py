import en_core_web_sm

parser = en_core_web_sm.load()
sentences = "There is an art, it says, or rather, a knack to flying." \
            "The knack lies in learning how to throw yourself at the ground and miss." \
            "In the beginning the Universe was created. This has made a lot of people " \
            "very angry and been widely regarded as a bad move."

print("解析文本中包含的句子：")
sents = [sent for sent in parser(sentences).sents]
for x in sents:
    print(x)
"""
There is an art, it says, or rather, a knack to flying.
The knack lies in learning how to throw yourself at the ground and miss.
In the beginning the Universe was created.
This has made a lot of people very angry and been widely regarded as a bad move.
"""
print("- * -"*20)
# 分词
print()
tokens = [token for token in sents[0] if len(token) > 1]
print(tokens)
print("- * -"*20)

# 词性还原
lemma_tokens = [token.lemma_ for token in sents[0] if len(token) > 1]

print(lemma_tokens)
print("- * -"*20)

# 简化版的词性标注
pos_tokens = [token.pos_ for token in sents[0] if len(token) > 1]
print(pos_tokens)
print("- * -"*20)

# 词性标注的细节版
tag_tokens = [token.tag_ for token in sents[0] if len(token) > 1]
print(tag_tokens)
print("- * -"*20)

# 依存分析
dep_tokens = [token.dep_ for token in sents[0] if len(token) > 1]
print(dep_tokens)
print("- * -"*20)


print("名词块分析")
doc = parser(u"Autonomous cars shift insurance liability toward manufacturers")
# 获取名词块文本
chunk_text = [chunk.text for chunk in doc.noun_chunks]
print(chunk_text)
print("- * -"*20)

# 获取名词块根结点的文本
chunk_root_text = [chunk.root.text for chunk in doc.noun_chunks]
print(chunk_root_text)
print("- * -"*20)

# 依存分析
chunk_root_dep_ = [chunk.root.dep_ for chunk in doc.noun_chunks]
print(chunk_root_dep_)
print("- * -"*20)
#
chunk_root_head_text = [chunk.root.head.text for chunk in doc.noun_chunks]
print(chunk_root_head_text)
print("- * -"*20)


