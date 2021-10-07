import sklearn
import pandas as pd
import numpy
import textblob
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('datamentah.csv')

df2 = df.drop_duplicates(ignore_index=True)
df3 = df2.astype(str)
df4 = df3.copy()
df4['Tweet'] = df3['Tweet'].replace({'"': '',
                             '\d+': '',
                             ':': '',
                             ';': '',
                             '#': '',
                             '@': '',
                             '_': '',
                             ',': '',
                             "'": '',
                             }, regex=True)
df4['Tweet'] = df4['Tweet'].str.replace(r'["\n"]+[https]+[?://]+[^\s<>"]+|www\.[^\s<>"]+[@?()]+[(??)]+[)*]+[(\xa0]+[-&gt...]', "",regex=True)
df4['Tweet'] = df4['Tweet'].str.replace('\n','',regex=True)
df4['Tweet'] = df4['Tweet'].str.replace('https//','',regex=True)
df4['Tweet'] = df4['Tweet'].str.replace('.','',regex=True)
df4['Tweet'] = df4['Tweet'].str.lstrip()
df4['Tweet'] = df4['Tweet'].str.lower()
df4['Tweet'] = df4['Tweet'].replace({'WTS':'','PT':'','RT':'','xa0PT':''},regex=True)

df5 = df4.copy()
df4 = df5[['Tweet']]


def getPolarity(text):
    return TextBlob(text).sentiment.subjectivity

def getSubjectivity(text):
    return TextBlob(text).sentiment.polarity


df4['Subjectivity'] = df4['Tweet'].apply(getSubjectivity)
df4['Polarity'] = df4['Tweet'].apply(getPolarity)

allwords = ''.join([twts for twts in df4['Tweet']])
wordCloud = WordCloud(width=500,height=300,random_state=21,max_font_size=100).generate(allwords)

plt.imshow(wordCloud,interpolation='bilinear')
plt.axis('off')
plt.show()

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df4['Analysis'] = df4['Polarity'].apply(getAnalysis)
'''
j = 1
sortedDF = df4.sort_values(by=['Polarity'])
for i in range (0,sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Positive'):
        print(str(j)+ ') '+sortedDF['Tweet'][i])
        print()
        j = j+1'''

import plotly.express as px
'''for i in range(0,df4.shape[0]):
    py.scatter(x=df4['Polarity'][i],y=df4['Subjectivity'][i])
fig.show()'''


for i in range(0,df4.shape[0]):
    plt.scatter(df4['Polarity'][i],df4['Subjectivity'][i],color='Red')
plt.grid()
plt.title('Sentimen Analysis')
plt.xlabel('Polarity')
plt.ylabel("Subjectivity")
plt.show()


ptweets = df4[df4.Analysis == 'Positive']
ptweetsc = ptweets.copy()
ptweets = ptweetsc[['Tweet']]
round((ptweets.shape[0] / df4.shape[0]) *100,1)

ntweets = df4[df4.Analysis == 'Negative']
ntweetsc = ntweets.copy()
ntweets = ntweetsc[['Tweet']]
round((ntweets.shape[0] / df4.shape[0]) *100,1)

nettweets = df4[df4.Analysis == 'Neutral']
netweetsc = nettweets.copy()
nettweets = nettweets[['Tweet']]
round((ntweets.shape[0] / df4.shape[0]) *100,1)


df4['Analysis'].value_counts()
plt.title('Sentimen Analysis')
plt.xlabel('Sentimen')
plt.ylabel('Counts')
df4['Analysis'].value_counts().plot(kind='bar')
plt.show()

