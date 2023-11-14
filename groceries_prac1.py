################################importing packages
from collections import Counter
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


############################### taking data
groceries=[]
with open("C:/Users/yamini/Desktop/GitHub/Association rules/groceries.csv") as f:
    groceries=f.read()
    
groceries            #it is in the form of text with \n delimiter

################################# Data preprocessing
#converting text to list using \n delimiter
groceries_list= groceries.split("\n")    #automatically converts into list.
#stil datatype is in string format. So we have to convert it to list

#converting datatype to list using "," delimiter
groceries_list1=[]

for i in groceries_list:
    groceries_list1.append(i.split(","))

groceries_list1             #now data type becomes list

#checking frequencies for each item
#keeping each item into seperate rows
groceries_freq=[i for item in groceries_list1 for i in item]
groceries_freq

groceries_freq1=Counter(groceries_freq)
groceries_freq1

#sorting groceries_freq1 to know which item is mostly sold
groceries_freq1_sort_asc=sorted(groceries_freq1.items(), key=lambda x:x[1], reverse=True)
groceries_freq1_sort_asc
#milk and vegetables are highly sold

####################################### apriori for support
#converting groceries_list1 into dataframe
groceries_df=pd.DataFrame(pd.Series(groceries_list1))

#changing dataframe column into "transactions"
groceries_df.columns=["transactions"]

#taking dummy variables
groceries_df1=groceries_df["transactions"].str.join(sep="*").str.get_dummies(sep="*")

#apriori
groceries_apriori=apriori(groceries_df1,min_support=0.0075, max_len=4, use_colnames=True)

#checking which item is having maximum support
groceries_apriori.sort_values('support', ascending=False, inplace=True )
groceries_apriori
#milk is having more support

#association rules
groceries_ass=association_rules(groceries_apriori, metric="lift", min_threshold=1)
groceries_ass.sort_values('lift', ascending=False, inplace=True)
groceries_ass
#lift ratio of yogurt and vegetables are high. When yogurt is sold, vegetables selling ratio is high
#tropical fruit and milk is next high lift ratio. If tropical fruits are selling, then there is 
#high ratio in selling of milk.











