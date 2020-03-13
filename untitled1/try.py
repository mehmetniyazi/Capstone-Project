import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import numpy as np
from mlxtend.frequent_patterns import association_rules



def reachfive(dataframe,item):# this is the function that data analysis done  . İt take dataframe and string value
    y=dataframe[dataframe['antecedents'] == {item}]
    print(y[['antecedents', 'consequents', 'support','confidence']].sort_values(by=['confidence','support'],ascending=False))
    return



data=pd.read_csv('Market_Basket_Optimisation.csv',header=None) #Loading data

yy=data.to_numpy()
ded=[]

for zz in yy:
    zz = zz[np.logical_not(pd.isnull(zz))]# with this i delete the NaN values
    ded.append(zz.tolist())
listofitems=[]
for i in ded:
    for ii in i:
        if ii not in listofitems:
            listofitems.append(ii) # with this i collect the item names in one array
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt','Mehmet'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs','Mehmet'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt','Mehmet'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs','Mehmet']] # Test data
te=TransactionEncoder()
te_ary=te.fit(ded).transform(ded)
df=pd.DataFrame(te_ary,columns=te.columns_) #With this three line of code i create the datafame available for apriori lagoirthm
#print(df)
x=apriori(df,min_support=0.0023,use_colnames=True) # With this i create the dframe for assocaitin rule. The lower value of min_support getting error




association_rules(x, metric="confidence", min_threshold=0.0000007)
rules=association_rules(x, metric="lift", min_threshold=0.00002)


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules[ (rules['antecedent_len'] >= 2) & (rules['confidence'] > 0.00000705) &(rules['lift'] > 0.00002) ] # with these code we create association rules with the value of confidence and lift
count=0

#x['length'] = x['itemsets'].apply(lambda x: len(x))            #test code
#print(x[ (x['length'] >= 2) &(x['support'] >= 0.8) ])          #test code
#print(x[ x['itemsets'] == frozenset(('Eggs', 'Onion'))  ])     #test code
#association_rules(x, metric="confidence", min_threshold=0.7)   #test code
#rules = association_rules(x, metric="lift", min_threshold=1.2) #test code
#print(rules)  #test code                                       #test code
#print(rules.columns.values)                                     #test code
#rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))     #test code
#print(rules)                                                   #test code
#print(rules[ rules['antecedents'] == {'Onion', 'Kidney Beans'}  ]) #test code
#print(rules.columns.values)                                    #test code
#y=rules[rules['antecedents'] == {listofitems[0]}]              #test code
#print(y[['antecedents', 'consequents', 'support','confidence']].sort_values(by=['confidence','support'],ascending=False))      #test code

reachfive(rules, listofitems[23])    #i call the data analysis funtion. It works with dataframe taken from assocation rules and string value




