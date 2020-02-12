# library for plotting

library(ggplot2)

# change address to location where the data files are - also assuming that the index.csv file is present

location = "E:\\iu\\fa18\\b555\\pp04\\pp4data\\20newsgroups"

# contains functions for logistic regression place this file in the same place as the code and enter the address

source("E:\\iu\\fa18\\b555\\pp04\\basic_functions.R")

set.seed(7)

##################################### TASK 1 #####################################

setwd(location)
files = list.files(location)
files = files[-which(files=="index.csv")]
doc.dat.orig = data.frame(matrix(ncol = 2, nrow = 0))
names(doc.dat.orig) = c("word", "doc")

for(i in files)
{
  word = readLines(i, warn = FALSE)
  word = strsplit(word, split = " ")
  word = unlist(word)
  doc = rep(i, length(word))
  temp = cbind(word, doc)
  doc.dat.orig = rbind(doc.dat.orig, temp)
}

# specify the number of topics here

K = 20

# randomly assign topics

doc.dat.orig$topic = replicate(dim(doc.dat.orig)[1], sample(1:K, size = 1))

doc.dat = data.frame("word" = as.numeric(doc.dat.orig$word), "doc" = as.numeric(doc.dat.orig$doc), "topic" = as.numeric(doc.dat.orig$topic))

# specify alpha, beta, number of iterations here

alpha = 5/K
beta = 0.01
Nwords = dim(doc.dat)[1]
Niters = 500

V = length(unique(doc.dat$word))
D = length(unique(doc.dat$doc))

pn = sample(1:Nwords)
P = rep(0, K)
word.list = doc.dat$word[pn]
topic.list = doc.dat$topic[pn]
doc.list = doc.dat$doc[pn]
Cd = table(doc.list, topic.list)
nam.row.d = rownames(Cd)
nam.col.d = colnames(Cd)
Cd = as.matrix.data.frame(Cd)
Ct = table(topic.list, word.list)
nam.row.t = rownames(Ct)
nam.col.t = colnames(Ct)
Ct = as.matrix.data.frame(Ct)

# collapsed gibbs sampler

for(i in 1:Niters)
{
  for(n in 1:Nwords)
  {
    word = word.list[n]
    topic = topic.list[n]
    doc = doc.list[n]
    Cd[doc, topic] = Cd[doc, topic] - 1
    Ct[topic, word] = Ct[topic, word] - 1
    
    for(k in 1:K)
    {
      p1 = (Ct[k, word] + beta)/(V * beta + sum(Ct[k, ]))
      p2 = (Cd[doc, k] + alpha)/(K * alpha + sum(Cd[doc, ]))
      P[k] = p1 * p2
    }
    
    P = P/sum(P)
    topic = sample(1:K, size = 1, prob = P)
    topic.list[n] = topic
    Cd[doc, topic] = Cd[doc, topic] + 1
    Ct[topic, word] = Ct[topic, word] + 1
  }
}

rownames(Cd) = nam.row.d
colnames(Cd) = nam.col.d
rownames(Ct) = nam.row.t
colnames(Ct) = nam.col.t

# print out top 5 words for each topic

L = levels(doc.dat.orig$word)
top.5 = list()
for(i in 1:K)
{
  temp = sort(Ct[i,], decreasing = TRUE)
  temp = temp[1:5]
  names(temp) = L[as.numeric(names(temp))]
  top.5[[i]] = temp
}

# print top 5 words per topic

top.5

# topic representation vectors

doc.unique = unique(doc.dat.orig$doc)
topic.rep = matrix(ncol = K, nrow = D)
vec = rep(NA, K)

for(i in 1:D)
{
  doc = as.numeric(doc.unique[i])
  
  for(k in 1:K)
  {
    vec[k] = (Cd[doc, k] + alpha)/(K * alpha + sum(Cd[doc, ]))
  }
  
  topic.rep[i, ] = vec
}


##################################### TASK 2 #####################################

labels = read.csv("index.csv", header = FALSE)
names(labels) = c("doc", "class")
labels$doc = as.factor(labels$doc)
labels.doc = labels[sapply(doc.unique, FUN = function(x){match(x, labels$doc)}), "class"]

word.unique = unique(doc.dat.orig$word)
bag.words = matrix(ncol = length(word.unique), nrow = D)

count.words = function(x, y)
{
  sum((y == x))
}

for(i in 1:D)
{
  temp = doc.dat.orig[doc.dat.orig$doc == doc.unique[i], "word"]
  bag.words[i, ] = sapply(word.unique, FUN = count.words, y = temp)
}


# function to answer task 1 - sample and plot accuracies

plot_errors = function(t, X, sz)
{
  t = as.matrix(t)
  s = ceiling(dim(X)[1]/3)
  df_errors = data.frame(matrix(c(0, 0, 0), nrow = 1))
  colnames(df_errors) = c("iteration", "size", "error_logit")
  
  for(i in 1:30)
  {
    rand = sample(1:dim(X)[1], s, replace = FALSE)
    train_X = X[-rand, ]
    train_t = t[-rand, ]
    test_X = X[rand, ]
    test_t =  t[rand, ]
    
    size = seq(sz, dim(train_X)[1], sz)
    size[length(size)] = size[length(size)] + dim(train_X)[1] %% sz
    
    for(j in size)
    {
      w = logistic_model(train_t[1:j], train_X[1:j, ], 0.01)[[1]]
      H = logistic_model(train_t[1:j], train_X[1:j, ], 0.01)[[2]]
      pred_logit = logistic_test(test_X, w, H, test_t)[[1]]
      err_logit = ifelse(pred_logit != test_t, 0, 1)
      df_errors = rbind(df_errors, c(i, j, mean(err_logit)))
    }
  }
  
  df_errors = df_errors[-1, ]
  return(list(aggregate(error_logit ~ size, FUN = mean, data = df_errors), aggregate(error_logit ~ size, FUN = sd, data = df_errors)))
}

# run till it says plot 1 for generating plot

df1 = plot_errors(labels.doc, topic.rep, 10)
dfa = df1[[1]]
dfb = df1[[2]]
names(dfb) = c("size", "sd_logit")
m1 = cbind(dfa, dfb)
m1 = m1[, -1]

df1 = plot_errors(labels.doc, bag.words, 10)
dfa = df1[[1]]
dfb = df1[[2]]
names(dfb) = c("size", "sd_logit")
m2 = cbind(dfa, dfb)
m2 = m2[, -1]

p = ggplot(m2, aes(x = size, y = error_logit, color = "bag of words")) + geom_point() + geom_line() + geom_errorbar(aes(ymin = error_logit + sd_logit, ymax = error_logit - sd_logit))
p = p + geom_point(data = m1, aes(x = size, y = error_logit, color = "topic representation")) + geom_line(data = m1, aes(x = size, y = error_logit, color = "topic representation")) + geom_errorbar(data = m1, aes(ymin = error_logit + sd_logit, ymax = error_logit - sd_logit, color = "topic representation"))
p + ggtitle("learning curves") + ylab("accuracy")

# plot 1

