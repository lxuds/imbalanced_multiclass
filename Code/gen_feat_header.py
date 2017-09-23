# -*- coding: <coding> -*-
import re
import pandas as pd
#from textacy import preprocess_text

df = pd.read_csv('./data/Interview_Mapping.csv')

'''
from collections import Counter
c = Counter()
l = np.array([1, 2, 1, 4, 1])
for i in range(len(l)):
   c[i]=l[i]


t = np.array(list(c.elements()))

ll = np.random.choice(l, size=l.shape, replace=True)
'''


names =[]
# No.DoJ l =[19, 30, 65, 71, 121, 174, 289, 300, 306, 431, 470, 581, 606, 621, 711, 727, 795, 800, 906, 946, 948, 953]
for i in range(df.shape[0]):
#for j in range(0,48):
#    i = l[j]
    f = open('./data/FixedJudgements/'+ df.iloc[i,0] + '.txt')
    raw = f.read()
    text1 = raw.split('ADVOCATES')[0]
    text2 = text1.split('Judges')[1]
    #print text2
    #pattern = r'^([a-z\. ]+) [0-9]'
    #p1 = r'^([a-z\. ]+) (?:[0-9])'
    pattern = r'^([a-z][a-z-\. ]+)(?:[0-9]|Case|Appeal|\()'
    obj = re.findall(pattern, text2, re.I|re.MULTILINE)
    #print i, df.iloc[i,0],  obj
    #tmp = "ADVOCATES" in raw
    #tmp = "Judges" in raw
    #print tmp
    names.append(obj)
#print names

clean_name = []
for i, x in enumerate(names):
    if x:
       a = x[0].upper()
       a = a.replace('APPEAL NO.', '') 
       a = a.replace('NO.', '') 
       a = a.replace(' NOS.', '') 
       a = a.replace('CASE', '')
       a = a.replace('NO.DOJ', '')
       a = a.replace('DOJ', '')
       a = a.replace('CIV. RULE', 'CIVIL')
       a = a.replace('I.-T. REF', 'INCOME TAX')
       #a = re.sub(r'C. R.', 'C.R.', a )
       #a = re.sub(r'C.R\z', 'C.R.', a )
       #a = a.replace('C. R.', 'C.R.')
       #a = a.replace('CR', 'C.R.')
       a = a.replace('A. O. D.', 'A.O.D.')
       a = a.replace('S. A. W O.', 'S.A.W.O.')
       a = a.rstrip()
       if 'INCOME' in a: 
           a = 'INCOME TAX'
       if 'CIVIL' in a or a =='CIV. REVN.':
           a = 'CIVIL'
       if 'CRIMINAL' in a or a =='CR. REVN.'or a =='CR. M.'or a =='CRL.A.' or a =='CRI.REVN.PETN' or a =='CR.MISC.' or a =='CRI MISC.' or a =='CRI.M.' or a =='CRI. REVISION' or a =='CRL. M.' or a =='CR. REV' or a =='CR. REV.' or a =='CRI. REV.' or a =='CRL W.' or a =='CRI.W.' or a =='CR. R.': 
           a= 'CRIMINAL' 
       if 'MATTER' in a:
           a = 'MATTER'
       if a == 'C. O.' or a =='C.O' or a =='CO.' or a =='CO': a = 'C.O.'
       if a =='C. R.' or a =='C.R' or a =='CR' or 'C.R. ' in a or a=='CR. APP.' or a == 'CR.' : a ='C.R.'
       if a =='A.F.O.O' or a =='AFOO': a ='A.F.O.O.'
       if a =='AFAD': a ='A.F.A.D.'
       if a =='S. A.' or a =='SA': a ='S.A.'
       if a =='F. A.': a ='F.A.'
       if a =='F.M.A' or a =='F. M. A.' or a =='FMA': a ='F.M.A.'
       if a =='AFOD': a ='A.F.O.D.'
    if not x:
       a = ""
    clean_name.append(a)
    #print i,  df["Judgements"].iloc[i], a
df['source'] = clean_name
b = df['source'].value_counts()
b.to_csv('a.csv', index=True)
#idn =[i for i in range(len(clean_name)) if clean_name[i] =='No.DOJ']
#print idn
#print df["Judgements"].iloc[idn]
df_test = df.loc[df['Area.of.Law'] =='To be Tested']

df_train = df.loc[df['Area.of.Law'] !='To be Tested']

Y_train = df_train['Area.of.Law']

#print pd.crosstab(df_train['Area.of.Law'], df_train['source'])
groupby_source = df_train['source'].groupby(df_train['Area.of.Law'])
print groupby_source.describe()

print df_train.info()
#print all(names)
#l =[i for i in range(df.shape[0]) if not names[i]]
#print len(l)
#print l
'''    
    #HON?BLE
    #print text1
    pattern = r'(?:BLE|CHIEF) (?:JUSTICE M(?:R|S|RS).|M(?:R|S|RS). JUSTICE) ([a-z\. ]+)(?:,|&|\n|A.F.O.D.)'

    obj = re.findall(pattern, text1, re.I)
    print i, df_train.iloc[i,0],  obj
    obj = [x.rstrip() for x in obj]
    #pattern1 = r'([A-Z]\.)*([A-Z]*)'
    #pattern_last_name = r'[ \.]([A-Z]*)$'
    #obj = re.findall(pattern_last_name, obj[1], re.I)
    pattern1 = r'([A-Z]\.)([A-Z]\.)([A-Z]*)'
    obj = re.match(pattern1, obj[0], re.I)
    print obj.group()
    print obj.group(1)+obj.group(2) + ' ' + obj.group(3)
    print ''.join([obj.group(1), obj.group(2), ' ', obj.group(3)])
    names.append(obj)
    #pattern = r'(?:BLE|CHIEF) (?:JUSTICE M[R|S|RS].|M[R|S|RS]. JUSTICE) ([a-z\. ]+)[,|&|\n]'
'''
#df_train['judges']= names
#df_train.info()
#print df_train.head()

#([A-Z]\.)+([A-Z]*)



'''
names =[]
for i in range(df_train.shape[0]):
#for i in range(868, 869):
    f = open('./data/FixedJudgements/'+ df_train.iloc[i,0] + '.txt')
    raw = f.read()
    text1 =  raw.split('Judgment')[0]
    #HON?BLE
    #print text1
    pattern = r'(?:BLE|CHIEF) (?:JUSTICE M(?:R|S|RS).|M(?:R|S|RS). JUSTICE) ([a-z\. ]+)(?:,|&|\n|A.F.O.D.)'

    obj = re.findall(pattern, text1, re.I)
    print i, df_train.iloc[i,0],  obj
    obj = [x.rstrip() for x in obj] 
    #pattern1 = r'([A-Z]\.)*([A-Z]*)'    
    #pattern_last_name = r'[ \.]([A-Z]*)$'
    #obj = re.findall(pattern_last_name, obj[1], re.I)
    pattern1 = r'([A-Z]\.)([A-Z]\.)([A-Z]*)'
    obj = re.match(pattern1, obj[0], re.I)
    print obj.group()
    print obj.group(1)+obj.group(2) + ' ' + obj.group(3)
    print ''.join([obj.group(1), obj.group(2), ' ', obj.group(3)])
    names.append(obj)
    #pattern = r'(?:BLE|CHIEF) (?:JUSTICE M[R|S|RS].|M[R|S|RS]. JUSTICE) ([a-z\. ]+)[,|&|\n]'

#df_train['judges']= names
#df_train.info()
#print df_train.head()

#([A-Z]\.)+([A-Z]*)


'''

'''
names_judgment =[]
#for i in range(df_train.shape[0]):
for i in range(331, 332):
    f = open('./data/FixedJudgements/'+ df_train.iloc[i,0] + '.txt')
    raw = f.read()
    text1 =  raw.split('Judgment')[1]
    text2 = text1.splitlines()
    #text2 = text1.split('\n')[0]
    #HON?BLE
    #print text1
    print 'text2[0]', text2[0]
    print 'text2[1]', text2[1]
    #print text2[2:10]
    pattern = r'^([a-z\. ]+)[, .]+J.'
    if text2[0]: 
        judge_text = text2[0]
    else:
       judge_text = text2[1]
    obj = re.findall(pattern, judge_text, re.I)
    print i, df_train.iloc[i,0],  obj, judge_text
    #names.append(obj)
'''
'''

#print df_train.index[df_train.isnull().any(axis=1)]

#print [i for i, x in enumerate(names) if not x ]


f = open('./data/FixedJudgements/'+ df_train.iloc[0,0] + '.txt')
raw = f.read()
pattern = r'THE HONOURABLE [CHIEF] (?:JUSTICE MR.|MR. JUSTICE) ([a-z\. ]+)[,|&|\n]'
obj = re.findall(pattern, raw, re.I)
print obj



def drop_html(html):
    return BeautifulSoup(html,"lxml").get_text(separator=" ",strip=True)


raw = drop_html(raw)


my_list = raw.splitlines()


pattern = r'JUSTICE MR.|MR. JUSTICE ([a-z.\s]+)'
re.findall(pattern, raw)


fout = 'vocab.txt'
fo = open(fout, "w")
for k, v in vec_tfidf.vocabulary_.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')




fo.close()

x_train2[0]



plt.hist(Y_train)
plt.show()
plt.savefig("foo.png")
plt.close()
'''

