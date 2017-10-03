# -*- coding: <coding> -*-
import re
import pandas as pd
#from textacy import preprocess_text
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('./data/Interview_Mapping.csv')
df_test = df.loc[df['Area.of.Law'] =='To be Tested']
df_train = df.loc[df['Area.of.Law'] !='To be Tested']

ind_test = df.index[df['Area.of.Law'] =='To be Tested']
ind_train = df.index[df['Area.of.Law'] !='To be Tested']
print ind_test




#############################
# extract category indicator
names =[]
for i in range(df.shape[0]):
    f = open('./data/FixedJudgements/'+ df.iloc[i,0] + '.txt')
    raw = f.read()
    text1 = raw.split('ADVOCATES')[0]
    text2 = text1.split('Judges')[1]
    pattern = r'^([a-z][a-z-\. ]+)(?:[0-9]|Case|Appeal|\()'
    obj = re.findall(pattern, text2, re.I|re.MULTILINE)
    names.append(obj)

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
       a = a.replace('A. O. D.', 'A.O.D.')
       a = a.replace('S. A. W O.', 'S.A.W.O.')
       a = a.rstrip()
       if 'INCOME' in a or a=='I.T.REF.' or a=='I.T.R.' or a=='I.T. REF.': 
           a = 'INCOME TAX'
       if 'CIVIL' in a or a =='CIV. REVN.':
           a = 'CIVIL'
       if 'CRIMINAL' in a or a =='CR. REVN.'or a =='CR. M.'or a =='CRL.A.' or a =='CRI.REVN.PETN' or a =='CR.MISC.' or a =='CRI MISC.' or a =='CRI.M.' or a =='CRI. REVISION' or a =='CRL. M.' or a =='CR. REV' or a =='CR. REV.' or a =='CRI. REV.' or a =='CRL W.' or a =='CRI.W.' or a =='CR. R.': 
           a= 'CRIMINAL' 
       if 'MATTER' in a:
           a = 'MATTER'
       if 'COMPANY' in a:
           a = 'COMPANY'
       if a == 'C. O.' or a =='C.O' or a =='CO.' or a =='CO': a = 'C.O.'
       if a =='C. R.' or a =='C.R' or a =='CR' or 'C.R. ' in a or a=='CR. APP.' or a == 'CR.' : a ='C.R.'
       if a =='A.F.O.O' or a =='AFOO' or a =='APPEAL FROM ORIGINAL ORDER': a ='A.F.O.O.'
       if a =='AFAD' or a =='APPEAL FROM APPELLATE DECREE': a ='A.F.A.D.'
       if a =='APPEAL FROM APPELLATE ORDER'or a=='APPEAL FROM APPELLATE ORDERS': a='A.F.A.O.'
       if a =='S. A.' or a =='SA': a ='S.A.'
       if a =='F. A.' or a=='FIRST APPEAL ORDER' or a =='FIRST APPEAL': a ='F.A.'
       if a =='F.M.A' or a =='F. M. A.' or a =='FMA': a ='F.M.A.'
       if a =='F.M.A.T' or a=='FMAT': a ='F.M.A.T.'
       if a =='AFOD' or a =='APPEAL FROM ORIGINAL DECREE': a ='A.F.O.D.'
       if 'FULL BENCH REF' in a: a ='FULL BENCH REF.'
       if 'LETTERS PATENT' in a or a =='L.P.A.': a= 'LETTERS PATENT'
       if 'SUPREME' in a: 
           a ='SUPREME COURT' 
       if a=='ORIGINAL DECREE': a='O.D.'
       if a =='APPEALS': a='APPEAL'
    if not x:
       a = ""
    clean_name.append(a)
    #print i, a, ' -> ', df['Area.of.Law'][i]
    #print i,  df["Judgements"].iloc[i], a
df['source'] = clean_name
b = df['source'].value_counts()
b.to_csv('a.csv', index=True)


# combine categories with case < 2
threshold = 2
value_counts = df['source'].value_counts() # Specific column 
#print value_counts
#print dir(value_counts)
to_remove = value_counts[value_counts < threshold].index
df['source'].replace(to_remove, 'OTHER', inplace=True)
df['source'].replace('', 'MISSING', inplace=True)

b = df['source'].value_counts()
#print b

#source_dummies = pd.get_dummies(df['source'])


mlb = MultiLabelBinarizer()
source_dummies = mlb.fit_transform([[x] for x in df['source']])
#print mlb.classes_
# 34 features
#print len(a[0])
#print a[0]
print source_dummies.shape

source_dummies_test = source_dummies[ind_test]
source_dummies_train = source_dummies[ind_train]

print source_dummies_test.shape
print source_dummies_train.shape

#print df.groupby(['Area.of.Law','source']).size() #.order(ascending=False)
#df = df.loc[df['Area.of.Law'] !='To be Tested']
for  name, group in df.groupby(['Area.of.Law']):
   # print name
   # print (group)
    #newdf = process(group)
    with open('group.csv', 'a') as f:
        group.to_csv(f)





'''
df_test = df.loc[df['Area.of.Law'] =='To be Tested']

df_train = df.loc[df['Area.of.Law'] !='To be Tested']

Y_train = df_train['Area.of.Law']


#df_train = df.loc[df['Area.of.Law'] !='To be Tested']
'''
# extract names
from collections import defaultdict
dd = defaultdict(list)
names = []
for i in range(df.shape[0]):
    f = open('./data/FixedJudgements/'+ df.iloc[i,0] + '.txt')
    raw = f.read()
    text1 =  raw.split('Judgment')[0]
    #text1 =text1.replace(',', '.')
    pattern = r'(?:JUSTICE M(?:R|S|RS).|M(?:R|S|RS). JUSTICE) ([a-z\. ]+)(?:,|&|\n|A.F.O.D.)'    
    #pattern = r'(?:BLE|CHIEF) (?:JUSTICE M(?:R|S|RS).|M(?:R|S|RS). JUSTICE) ([a-z\. ]+)(?:,|&|\n|A.F.O.D.)'
    #pattern1 = r'ABLE M(?:R|S|RS). JUSTICE ([a-z\., ]+)(?:,|&|\n|A.F.O.D.)' 
    #pattern2 = r'HONOURABLE CHIEF JUSTICE M(?:R|S|RS). ([a-z\., ]+)(?:,|&|\n|A.F.O.D.)'
    #obj = re.findall('|'.join([pattern1,pattern2]), text1, re.I)
    obj = re.findall(pattern, text1, re.I)
    obj = [x.rstrip() for x in obj]

    names.append(obj)
    #print  i, df.iloc[i,0], obj 


# extract last name
last_names =[]
for i in range(df.shape[0]):
    obj = names[i]
    confirmed_last_name_list = []
    for name in obj:
        if '.' in name:
            pattern =r'((\w\.){1,}) *([A-Z ]{2,})'
            obj_new = re.findall(pattern, name, re.I)
            if obj_new:
                last_name = obj_new[0][2]
                if last_name not in last_names:
                    last_names.append(last_name)
                confirmed_last_name_list.append(last_name)
               

    #print i, df.iloc[i,0], obj, confirmed_last_name_list 
#print last_names
#print len(set(last_names))

# extract full name, and abbreviate first and mid names
abb_names = []
abb_names_list = []
for i in range(df.shape[0]):
#for i in [741, 742]:
    obj = names[i]
    abb_name_list_each = []
    for name in obj:
        name = ' '.join(name.split()).upper()
        if '.' in name:
            pattern =r'(\w{1,}) (\w\.){1,} *([A-Z ]{2,})'
            obj_new = re.findall(pattern, name, re.I)
            if obj_new:
                new_name = obj_new[0][0][0] + '.' + obj_new[0][1] + ' ' + obj_new[0][-1]
            else:
                pattern =r'((\w\. *){1,}) *([A-Z ]{2,})'
                obj_new = re.findall(pattern, name, re.I)
                if obj_new:
                    new_name = ''.join(obj_new[0][0].split()) + ' ' + obj_new[0][2]
        else:
            if name in last_names:
                new_name = name
            else: 
                a_list = name.split(' ')
                if len(a_list) == 1:
                    new_name = name
                else:
                    first_l = [a_list[j][0] for j in range(0,len(a_list)-1) ]
                    new_name = '.'.join(first_l+ [' ']) + a_list[-1]
        abb_name_list_each.append(new_name)
        if new_name not in abb_names:
            abb_names.append(new_name)
        if name not in dd[new_name]:
            dd[new_name].append(name)
    #print  i, df.iloc[i,0], obj, abb_name_list_each
    abb_names_list.append(abb_name_list_each)
 

a= sorted(dd.iteritems(),key=lambda (k,v):v[0])

#for i, x in enumerate(a):
#  print i, x


#print i, df.iloc[i,0], names[i], abb_name_list_each, abb_names_list[i]
#print len(abb_names)
#print len(abb_names_list)


df['abb_names'] = abb_names_list

mlb = MultiLabelBinarizer()
abb_names_dummies = mlb.fit_transform(abb_names_list)

abb_names_dummies_test = abb_names_dummies[ind_test]
abb_names_dummies_train = abb_names_dummies[ind_train]
#print mlb.classes_
#print len(mlb.classes_)
#print abb_names_dummies[0]

print abb_names_dummies_test.shape
print abb_names_dummies_train.shape


'''
from itertools import compress
print list(compress(mlb.classes_, abb_names_dummies[0]))
print abb_names_list[0]
'''


# extract court location
court_loc_list = []
court_loc_all = []
#court_all = []
for i in range(df.shape[0]):
    f = open('./data/FixedJudgements/'+ df.iloc[i,0] + '.txt')
    raw = f.read()
    text1 =  raw.split('Judges')[0]
    pattern = r'^(\w+) Court (?:.*? at|of) (\w+)'
    obj = re.search(pattern, text1, re.I|re.MULTILINE)
    if obj:
       court_loc = obj.group(2)
       court = obj.group(1)
       #print i, df.iloc[i,0], court, court_loc
       if court_loc not in court_loc_all:
           court_loc_all.append(court_loc)
       #if court not in court_all:
       #    court_all.append(court)
    else:
        court_loc = ''
    court_loc_list.append(court_loc)
    #print i, df.iloc[i,0], obj,
#print court_loc_all
#print court_loc_list
df['court_loc'] = court_loc_list
print df.loc[df['court_loc'] =='' ]

print df['court_loc'].value_counts()
mlb = MultiLabelBinarizer()
loc_dummies = mlb.fit_transform([[x] for x in court_loc_list])
print mlb.classes_
print loc_dummies[0]
loc_dummies_test = loc_dummies[ind_test]
loc_dummies_train = loc_dummies[ind_train]
#print mlb.classes_
#print len(mlb.classes_)
#print loc_dummies[0]

print loc_dummies_test.shape
print loc_dummies_train.shape



#print df_train['Area.of.Law'].groupby(df_train['court_loc']).describe()

#print df_train.groupby(['court_loc', 'Area.of.Law']).size()

