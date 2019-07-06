# ������Ҫ��packages
library(readxl)
library(jiebaR)# �������ݴ��·��
setwd("D:\\alvin_py\\business\\p201906\\D062505_450")
# ������������
evaluation <- read.csv("��.csv",quote = "",sep = ",", header = T, stringsAsFactors = F)
evaluation <- unique(evaluation$�ֶ�2) # ȥ��

# ��ȡ����͸���ʻ�
pos <- read.csv("����.txt", header = F, sep = ",", encoding = 'gbk')
neg <- read.csv("����.txt", header = F, sep = ",", encoding = 'gbk')
# ת�����ݸ�ʽ��תΪ������ʽ
pos_dic <- c(t(unique(pos)))
neg_dic <- c(t(unique(neg)))
mydict <- c(unique(pos), unique(neg))

engine <- worker()
# �����Զ���ʻ�
# ��������ʼ��뵽�Զ���ʿ���
new_user_word(engine, neg_dic)
new_user_word(engine, pos_dic)


# ��ÿһ�����۽����д�
segwords <- sapply(unique(evaluation), segment, engine)
head(segwords)
# ȥ��
fun <- function( x, y) x%in% y

# ������е÷ֺ���
getEmotionalType <- function( x,pwords,nwords){
  pos.weight = sapply(llply(x,fun,pwords),sum)  # ������дʵ�Ȩ�أ������÷�
  neg.weight = sapply(llply(x,fun,nwords),sum)
  total = pos.weight - neg.weight
  return(data.frame( pos.weight, neg.weight, total))
}
score <- getEmotionalType(segwords, pos, neg)
head(score)
evalu.score<- cbind(evaluation, score)  # �ѵ÷ֺ����ݺϲ�

# Ϊ��ÿ�����۴���������еı�ǩ���������ܵ÷�
# ���ڵ���0�ļ�¼����Ϊ������У�С��0�ļ�¼����Ϊ������С�
evalu.score <- transform(evalu.score, emotion = ifelse(total>= 0, 'pos', 'neg'))

write.csv(evalu.score, "evalu.score.csv")  # ���浽�ļ�
