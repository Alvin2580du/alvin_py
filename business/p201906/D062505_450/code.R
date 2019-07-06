# 加载需要的packages
library(readxl)
library(jiebaR)# 设置数据存放路径
setwd("D:\\alvin_py\\business\\p201906\\D062505_450")
# 读入评论数据
evaluation <- read.csv("总.csv",quote = "",sep = ",", header = T, stringsAsFactors = F)
evaluation <- unique(evaluation$字段2) # 去重

# 读取正面和负面词汇
pos <- read.csv("正面.txt", header = F, sep = ",", encoding = 'gbk')
neg <- read.csv("负面.txt", header = F, sep = ",", encoding = 'gbk')
# 转换数据格式，转为向量格式
pos_dic <- c(t(unique(pos)))
neg_dic <- c(t(unique(neg)))
mydict <- c(unique(pos), unique(neg))

engine <- worker()
# 添加自定义词汇
# 将正负面词加入到自定义词库中
new_user_word(engine, neg_dic)
new_user_word(engine, pos_dic)


# 对每一条评论进行切词
segwords <- sapply(unique(evaluation), segment, engine)
head(segwords)
# 去重
fun <- function( x, y) x%in% y

# 计算情感得分函数
getEmotionalType <- function( x,pwords,nwords){
  pos.weight = sapply(llply(x,fun,pwords),sum)  # 计算情感词的权重，给出得分
  neg.weight = sapply(llply(x,fun,nwords),sum)
  total = pos.weight - neg.weight
  return(data.frame( pos.weight, neg.weight, total))
}
score <- getEmotionalType(segwords, pos, neg)
head(score)
evalu.score<- cbind(evaluation, score)  # 把得分和数据合并

# 为给每个评论打上正负情感的标签，不妨将总得分
# 大于等于0的记录设置为正面情感，小于0的记录设置为负面情感。
evalu.score <- transform(evalu.score, emotion = ifelse(total>= 0, 'pos', 'neg'))

write.csv(evalu.score, "evalu.score.csv")  # 保存到文件

