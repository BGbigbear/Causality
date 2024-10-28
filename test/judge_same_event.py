import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


threshold=0.5

def extract_chinese(text):
    """保留字符串中的汉字字符"""
    return ''.join(re.findall(r'[\u4e00-\u9fff]+', text))

def get_keywords(text):
    """对文本分词并提取关键词"""
    words = jieba.lcut(text)
    return ' '.join(words)  # 返回分词后的字符串

def is_similar_event(s1, s2, threshold = threshold):
    """判断两个事件描述是否相似"""
    # 提取汉字字符
    s1 = extract_chinese(s1)
    s2 = extract_chinese(s2)

    # 提取关键词
    s1_keywords = get_keywords(s1)
    s2_keywords = get_keywords(s2)

    # 计算相似度
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([s1_keywords, s2_keywords])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity >= threshold

# # 示例
# s1 = "通用动力公司授予BAE系统公司一份舰炮合同，以为美海军第四艘濒海战斗舰提供装备。"
# s2 = "通用动力授予BAE公司舰炮合同，美海军第四艘战斗舰获得新装备。"
# result = is_similar_event(s1, s2)
# print("是否为相同事件:", result)
