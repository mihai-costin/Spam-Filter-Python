import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import snowballstemmer

# stemming 
stemmer = snowballstemmer.stemmer("english")
class TfIdfVect(TfidfVectorizer):
        def build_analyzer(self):
                analyzer = super(TfidfVectorizer, self).build_analyzer()
                return lambda doc: stemmer.stemWords(analyzer(doc))

# citirea fisierului csv
# fisierul contine 2654 de inregistrari
# dintre care 747 de mesaje sunt spam (28%), iar 1907 sunt ham in spamHam (72%)
f  = pd.read_csv("spamHam.csv", sep=",")

# schimb ham cu 1 si spam cu 0
f.loc[f["type"] == "ham", "type",] = 1
f.loc[f["type"] == "spam", "type",] = 0

# aleator
f = f.sample(frac=1)

# luam f_x ca fiind mesajul, iar f_y ca fiind status-ul (ham sau spam)
f_x = f["text"]
f_y = f["type"]

# initializare tfidf
# lowecase = True
# use_idf, smooth_idf, norm = {l1, ^l2}
# ngram_range = tuple
# max_features 
tfIdf = TfIdfVect(encoding="utf-8", stop_words="english", analyzer="word", 
                  smooth_idf = False, ngram_range = (1,2))

# separarea datelor in 80% pentru antrenare si 20% pentru test
x_train, x_test, y_train, y_test = train_test_split(f_x, f_y, stratify = f_y, test_size = 0.2)

# fit pe text cu scorul tfidf
x_trainTfIdf = tfIdf.fit_transform(x_train)

# alpha pentru Additive (Laplace/Lidstone) smoothing
# fit_prior - learn class prior probabilities or not. 
# naive bayes multinomial intializare
nbM  = MultinomialNB(alpha = 1.0)

# transform f_y, care contine 1 si 0 sub forma string, in intreg 
y_train = y_train.astype("int")

# Naive - Bayes
nbM.fit(x_trainTfIdf, y_train)

# transform status-ul textului intr-un array
a = np.array(y_test) 

# predictie  
x_testTfIdf = tfIdf.transform(x_test)
pred = nbM.predict(x_testTfIdf)

count = 0

for i in range(len(pred)):
    if a[i] == pred[i]:
        count = count + 1
        
countSpam = 0

for i in range(len(f_y)):
    if f_y[i] == 0:
        countSpam = countSpam + 1

print()
print("File: spamHam.csv")
print(" Fisierul contine", len(f_y), " inregistrari")
print(" Dintre care %d (%.2f %%) sunt mesaje spam, iar %d (%.2f %%) sunt ham" 
      %(countSpam, countSpam/len(f_y) * 100, len(f_y)-countSpam, (len(f_y)-countSpam)/len(f_y) * 100))
print(" Exacte ", count, " din", len(a), " (din testul oferit)")
print("Acuratete model: %.2f %%"  %(np.mean(pred == a) * 100))

# ------------------------
# predictie pe fisierul spamHam2.csv

df = pd.read_csv("spamHam2.csv", sep=",")

df.loc[df["type"] == "ham", "type",] = 1
df.loc[df["type"] == "spam", "type",] = 0

df = df.sample(frac=1)

df_tfidf = tfIdf.transform(df["text"])

pred2 = nbM.predict(df_tfidf)

b = np.array(df["type"])

count = 0

for i in range(len(pred2)):
    if pred2[i] == b[i]:
        count = count + 1

countSpam = 0

for i in range(len(df["type"])):
    if b[i] == 0:
        countSpam = countSpam + 1
        
print("-----------------------------------")
print("File: spamHam2.csv")
print(" Fisierul contine", len(b), " inregistrari")
print(" Dintre care %d (%.2f %%) sunt mesaje spam, iar %d (%.2f %%) sunt ham" 
      %(countSpam, countSpam/len(b) * 100, len(b)-countSpam, (len(b)-countSpam)/len(b) * 100))
print(" Exacte ", count, " din", len(b))
print(" Acuratete: %.2f %%" %(np.mean(pred2 == b) * 100))


print()
print(" In total s-au folosit ", len(f_y) + len(b), " inregistrari.")

# metrics - classification report

print()
print("Raport de clasificare asupra fisierului spamHam.csv:")
a1 = []
pred1 = []
for i in range(len(a)):
    a1.append(a[i])
    pred1.append(pred[i])
    
print(classification_report(a1, pred1))