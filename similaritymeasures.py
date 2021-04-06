
# coding: utf-8

# In[1]:



#!/usr/bin/env python

from math import*
from decimal import Decimal

class Similarity():

    """ Five similarity measures function """

    def euclidean_distance(self,x,y):

        """ return euclidean distance between two lists """

        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    def manhattan_distance(self,x,y):

        """ return manhattan distance between two lists """

        return sum(abs(a-b) for a,b in zip(x,y))

    def minkowski_distance(self,x,y,p_value):

        """ return minkowski distance between two lists """

        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

    def nth_root(self,value, n_root):

        """ returns the n_root of an value """

        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)

    def cosine_similarity(self,x,y):

        """ return cosine similarity between two lists """

        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return round(numerator/float(denominator),3)

    def square_rooted(self,x):

        """ return 3 rounded square rooted value """

    return round(sqrt(sum([a*a for a in x])),3)

    def jaccard_similarity(self,x,y):

        """ returns the jaccard similarity between two lists """

        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)
    
    def zeroCounter(x):
        i = 0
        for l in x:
            if l == 0:
                i += 1
        return i
    
    def Salton_ochiai(self,x,y):

        """ returns the Salton_ochiai similarity between two lists """

        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        lenX = len(x) - zeroCounter(x)
        lenY = len(y) - zeroCounter(y)
        multiply_cardinality = sqrt(lenX * lenY)
        # union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality / float(multiply_cardinality) 

def cosine_Salton_Ochiai(self, x, y, alpha=2.5):
    return (Salton_ochiai(self,x,y) ** alpha) * cosine_similarity(self,x,y)


# In[5]:


#!/usr/bin/env python

# from similaritymeasures import Similarity

def main():
   """ main function to create Similarity class instance and get use of it """
   measures = Similarity()
   print(measures.euclidean_distance([0,3,4,5],[7,6,3,-1]))
   
   print(measures.jaccard_similarity([0,1,2,5,6],[0,2,3,5,7,9]))

if __name__ == "__main__":
   main() 

