#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
from IPython.display import Image
import pydotplus
import utils
#plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns  # statistical data visualization with matplotlib
#stats
import math  
import scipy.stats


#Question1 :

def getPrior(df):
    """
    Calcule la probabilité a priori de la classe 1 ainsi que l'intervalle de
    confiance à 95% pour l'estimation de cette probabilité.
    *parametre df:
    Dataframe contenant les données. Doit contenir une colonne nommée "target" (contenont soit 0 soit 1 d'après la definition.
    *type de df:
    pandas dataframe
    *le return:
    de type Dictionnaire, contennant la moyenne et les extremités de l'intervalle de confiance. Clés 'estimation', 'min5pourcent', 'max5pourcent'.
    """
    result =  {}
    moy = df["target"].mean()
    tmp = 1.96 * math.sqrt((moy * (1 - moy))/df.shape[0]) # z=1.69 http://www.ltcconline.net/greenl/courses/201/estimation/smallConfLevelTable.htm
    result['estimation'] = moy
    result['min5pourcent'] = moy - tmp
    result['max5pourcent'] = moy + tmp
    
    return result


#Question2 :  programmation orientée objet dans la hiérarchie des Classifier

class APrioriClassifier (utils.AbstractClassifier):
    """
    Estime très simplement la classe de chaque individu par la classe majoritaire.
    """
    def __init__(self):
        pass
    
    def estimClass(self, attributs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        Pour ce APrioriClassifier, la classe vaut toujours 1.
        *parametre attributs :
        le  dictionnaire nom-valeur des attributs
        *le return :
        la classe 0 ou 1 estimée
        Pour ce APrioriClassifier, la classe vaut toujours 1.
        """
        return 1
    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.
        VP : nombre d'individus avec target=1 et classe prévue=1
        VN : nombre d'individus avec target=0 et classe prévue=0
        FP : nombre d'individus avec target=0 et classe prévue=1
        FN : nombre d'individus avec target=1 et classe prévue=0
        Précision : combien de candidats sélectionnés sont pertinents (VP/(VP+FP))
        Rappel : combien d'éléments pertinents sont sélectionnés (VP/(VP+FN))
        *le parametre df:
        le dataframe 
        *le return:
        un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """
        dictionnaire_proba = {}
        dictionnaire_proba["VP"] = 0
        dictionnaire_proba["VN"] = 0
        dictionnaire_proba["FP"] = 0
        dictionnaire_proba["FN"] = 0
        #calcul stats: estim représente l'estimation de la classe de la personne et  dic['target'] sa classe réelle
        for t in range(df.shape[0]):
            dic = utils.getNthDict(df, t)
            estim = self.estimClass(dic)
            if dic["target"] == 1:
                if estim == 1:
                    dictionnaire_proba["VP"]+= 1
                else:
                    dictionnaire_proba["FN"]+= 1
            else:
                if estim == 1:
                    dictionnaire_proba["FP"]+= 1
                else:
                    dictionnaire_proba["VN"]+= 1
                    
        dictionnaire_proba["Précision"] = dictionnaire_proba["VP"]/(dictionnaire_proba["VP"] + dictionnaire_proba["FP"])
        dictionnaire_proba["Rappel"] = dictionnaire_proba["VP"]/ (dictionnaire_proba["VP"] + dictionnaire_proba["FN"])
        
        return dictionnaire_proba
    
    
#Question3 : classification probabiliste à 2 dimensions
#Question3a :  probabilités conditionelles
    
def P2D_l(df, attr):
    """
    Calcul de la probabilité conditionnelle P(attribut | target).
    *les parametres: 
    df: dataframe avec les données. Doit contenir une colonne nommée "target".
    attr: attribut à utiliser, nom d'une colonne du dataframe.
    *le return:
    de type dictionnaire de dictionnaire, dictionnaire_proba. dictionnaire_proba[t][a] contient P(attribut = a | target = t).
    """
    #Valeurs possibles de l'attribut.
    list_cle = np.unique(df[attr].values) 
    dictionnaire_proba = {}
    #Target a toujours pour valeur soit 0 soit 1.
    dictionnaire_proba[0] = dict.fromkeys(list_cle, 0)
    dictionnaire_proba[1] = dict.fromkeys(list_cle, 0)
    
    group = df.groupby(["target", attr]).groups
    for t, val in group:
        dictionnaire_proba[t][val] = len(group[(t, val)])
    
    taille0 = (df["target"] == 0).sum()
    taille1 = (df["target"] == 1).sum()
    
    for i in list_cle:
        dictionnaire_proba[0][i] = dictionnaire_proba[0][i]/taille0
        dictionnaire_proba[1][i] = dictionnaire_proba[1][i]/taille1
        
    return dictionnaire_proba

def P2D_p(df, attr):
    """
    Calcul de la probabilité conditionnelle P(target | attribut).
    *les parametres: 
    df: dataframe avec les données. Doit contenir une colonne nommée "target".
    attr: attribut à utiliser, nom d'une colonne du dataframe.
    *le return:
    de type dictionnaire de dictionnaire, dictionnaire_proba. dictionnaire_proba[t][a] contient P(target = t | attribut = a).
    """
    list_cle = np.unique(df[attr].values) #Valeurs possibles de l'attribut.
    dictionnaire_proba = dict.fromkeys(list_cle)
    for cle in dictionnaire_proba:    
        dictionnaire_proba[cle] = dict.fromkeys([0,1], 0) #Target a toujours pour valeur soit 0 soit 1.
    
    group = df.groupby(["target", attr]).groups
    for t, val in group:
        dictionnaire_proba[val][t] = len(group[(t, val)])
        
    for cle in dictionnaire_proba:
        taille = (df[attr] == cle).sum()
        for i in range (2):
            dictionnaire_proba[cle][i] = dictionnaire_proba[cle][i] / taille
    return dictionnaire_proba


#Question3b :  classifieurs 2D par maximum de vraisemblance

class ML2DClassifier (APrioriClassifier):
    """
    Classifieur 2D par maximum de vraisemblance à partir d'une seule colonne du dataframe.
    """
    
    def __init__(self, df, attr):
        """
        *les parametres:
        df : dataframe.
        attr : le nom d'une colonne du dataframe df.
        """
        self.attr = attr
        self.dictionnaire_proba_p = P2D_l(df, attr)
        
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum de vraisemblance à partir de dictionnaire_proba_p.
        *le parametre attrs:
        le  dictionnaire nom-valeur des attributs
        *le return :
        la classe 0 ou 1 estimée
        """
        val = attrs[self.attr]
        #si la valeur de l'attribut n'existe pas dans l'ensemble d'apprentissage sa probabilité conditionnelle vaut zero.
        P = [0.0, 0.0]
        for t in [0, 1]:
            if val in self.dictionnaire_proba_p[t]:
                P[t] = self.dictionnaire_proba_p[t][val]
        if P[0] >= P[1]:
            return 0
        return 1
    
    
#Question3c : classifieurs 2D par maximum a posteriori

class MAP2DClassifier (APrioriClassifier):
    """
    Classifieur 2D par maximum a posteriori à partir d'une seule colonne du dataframe.
    """
    
    def __init__(self, df, attr):
        """
        *les parametres:
        df : dataframe.
        attr : le nom d'une colonne du dataframe df.
        """
        self.attr = attr
        self.dictionnaire_proba_p = P2D_p(df, attr)
        
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum a posteriori à partir de dictionnaire_proba_p.
        *le parametre attrs:
        le  dictionnaire nom-valeur des attributs
        *le return :
        la classe 0 ou 1 estimée
        """
        val = attrs[self.attr] 
        if val in self.dictionnaire_proba_p:
            if self.dictionnaire_proba_p[val][0] >= self.dictionnaire_proba_p[val][1]:
                return 0
            return 1
        else:
            #si la valeur de l'attribut n'existe pas dans l'ensemble d'apprentissage, sa probabilité conditionnelle n'est pas definie et on renvoie zero
            return 0
 
    
#Question4 :   
#Question4.1 : complexité en mémoire

def nbParams(df, liste = None):
    """
    Affiche la taille mémoire de tables P(target|attr1,..,attrk) étant donné un
    dataframe df et la liste [target,attr1,...,attrl], en supposant qu'un float
    est représenté sur 8 octets. Pour cela la fonction utilise la fonction 
    auxiliaire octetToStr().
    *les parametres: 
    df: Dataframe contenant les données. 
    liste: liste contenant les colonnes prises en considération pour le calcul. 
    """
    if liste is None:
        liste = list(df.columns) 
    taille = 8
    for col in liste:
        taille *= np.unique(df[col].values).size
    results = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        results = results + " = " + octetToStr(taille)
    print (results)  

def octetToStr(taille):
    """
    Transforme l’entier taille en une chaîne de caractèresults qui donne sa représentation
    en nombre d’octets, ko, mo, go et to. 
    *parametre taille: le nombre à être transformé.
    """
    suffixe = ["o", "ko", "mo", "go", "to"]
    results = ""
    for suf in suffixe:
        if taille == 0:
            break  
        if suf == "to":
            results = " {:d}".format(taille) + suf + results 
        else:
            results = " {:d}".format(taille % 1024) + suf + results
            taille //= 1024
    
    if results == "":
        results = " 0o"
    return results[1:]


#Question4.2 : complexité en mémoire sous hypothèse d'indépendance complète

def nbParamsIndep(df):
    """
    Affiche la taille mémoire nécessaire pour représenter les tables de probabilité
    étant donné un dataframe, en supposant l'indépendance des variables et qu'un
    float est représenté sur 8 octets.Pour cela la fonction utilise la fonction 
    auxiliaire octetToStr().
    
    *le parametre df: Dataframe contenant les données.  
    """
    taille = 0
    liste = list(df.columns) 
    
    for col in liste:
        taille += (np.unique(df[col].values).size * 8)
    
    results = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        results = results + " = " + octetToStr(taille)
    print (results)
    

#Question5 : Modèles graphiques
#Question5.3 :  modèle graphique et naïve bayes
    
def drawNaiveBayes(df, col):
    """
    Construit un graphe orienté représentant naïve Bayes.
    *les parametres:
    df: Dataframe contenant les données.  
    col: le nom de la colonne du Dataframe utilisée comme racine.
    *le return: 
    Le graphe.
    """
    tab_col = list(df.columns.values)
    tab_col.remove(col)
    resultant = ""
    for child in tab_col:
        resultant = resultant + col + "->" + child + ";"
    return utils.drawGraph(resultant[:-1])    

def nbParamsNaiveBayes(df, col_father, liste_col = None):
    """
    Affiche la taille mémoire de tables P(target), P(attr1|target),.., P(attrk|target) 
    étant donné un dataframe df, la colonne racine col_father et la liste [target,attr1,...,attrk],
    en supposant qu'un float est représenté sur 8 octets. Pour cela cette fonction 
    utilise aussi la fonction auxiliaire octetToStr().
    *Les parametres:
    df: Dataframe contenant les données. 
    col_father: le nom de la colonne du Dataframe utilisée comme racine.
    liste_col: liste contenant les colonnes prises en considération pour le calcul. 
    """
    taille = np.unique(df[col_father].values).size * 8
    
    if liste_col is None:
        liste_col = list(df.columns) 
        
    if liste_col != []:  
        liste_col.remove(col_father)
    
    for col in liste_col:
        temp = (np.unique(df[col_father].values).size * np.unique(df[col].values).size) * 8
        taille += temp
    
    results = str(len(liste_col)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        results = results + " = " + octetToStr(taille)
    print (results)
 
    
#Question5.4 :  classifier naïve bayes

class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes. 
    """
    def __init__(self, df):
        """
        Initialise le classifieur.
        *le parametre df: dataframe. 
        """
        self.dictionnaire_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            self.dictionnaire_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum de vraissemblance à partir de dictionnaire_res.
        *le paramtre attrs: le dictionnaire nom-valeur des attributs
        *le return:
        la classe 0 ou 1 estimée
        """
        dictionnaire_res = self.estimProbas(attrs)
        if dictionnaire_res[0] >= dictionnaire_res[1]:
            return 0
        return 1
        
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        *le parametre attrs: le dictionnaire nom-valeur des attributs
        """    
        P_0 = 1
        P_1 = 1
        for key in self.dictionnaire_P2D_l:
            dictionnaire_p = self.dictionnaire_P2D_l[key]
            if attrs[key] in dictionnaire_p[0]:
                P_0 *= dictionnaire_p[0][attrs[key]]
                P_1 *= dictionnaire_p[1][attrs[key]]
            else:
                return {0: 0.0, 1: 0.0}
        return {0: P_0, 1: P_1}
   
class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes. 
    """
    def __init__(self, df):
        """
        Initialise le classifieur.
        *le parametre df: dataframe. 
        """
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        self.dictionnaire_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            self.dictionnaire_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum à posteriori à partir de dictionnaire_res.
        *le paramtre attrs: le dictionnaire nom-valeur des attributs
        *le return:
        la classe 0 ou 1 estimée
        """
        dictionnaire_res = self.estimProbas(attrs)
        if dictionnaire_res[0] >= dictionnaire_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
        *le parametre attrs: le dictionnaire nom-valeur des attributs
        """    
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for key in self.dictionnaire_P2D_l:
            dictionnaire_p = self.dictionnaire_P2D_l[key]
            if attrs[key] in dictionnaire_p[0]:
                P_0 *= dictionnaire_p[0][attrs[key]]
                P_1 *= dictionnaire_p[1][attrs[key]]
            else:
                return {0: 0.0, 1: 0.0}
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}    


#Question6 : feature selection dans le cadre du classifier naive bayes
        
def isIndepFromTarget(df,attr,x):
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    
    *les parametres: 
    df : dataframe. 
    attr: le nom d'une colonne du dataframe df.
    x: seuil de confiance.
    """
    list_val = np.unique(df[attr].values) # Valeurs possibles de l'attribut.
    dictionnaire_val = {list_val[i]: i for i in range(list_val.size)} #association de chaque valeur a son indice en list_val dans un dictionnaire.
    matrix_cont = np.zeros((2, list_val.size), dtype = int)
    for i, row in df.iterrows():
        j =  row[attr]
        matrix_cont[row["target"], dictionnaire_val[j]]+= 1 
    _, p, _, _ = scipy.stats.chi2_contingency(matrix_cont) #on a utiliser scipy.stats.chi2_contingency après avoir étudier ce doc https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    return p > x

class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes reduit. 
    """
    def __init__(self, df, x):
        """
        Initialise le classifieur.
        *les parametres
        df: dataframe. 
        x: seuil de confiance. 
        """
        self.dictionnaire_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            if not isIndepFromTarget(df,attr,x):
                self.dictionnaire_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum de vraissemblance à partir de dictionnaire_res.
        *le paramtre attrs: le dictionnaire nom-valeur des attributs
        *le return:
        la classe 0 ou 1 estimée
        """
        dictionnaire_res = self.estimProbas(attrs)
        if dictionnaire_res[0] >= dictionnaire_res[1]:
            return 0
        return 1
        
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        *le parametre attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = 1
        P_1 = 1
        for key in self.dictionnaire_P2D_l:
            dictionnaire_p = self.dictionnaire_P2D_l[key]
            if attrs[key] in dictionnaire_p[0]:
                P_0 *= dictionnaire_p[0][attrs[key]]
                P_1 *= dictionnaire_p[1][attrs[key]]
            else:
                return {0: 0.0, 1: 0.0}
        return {0: P_0, 1: P_1}
    
    def draw(self):
        """
        Construit un graphe orienté représentant naïve Bayes réduit.
        """
        results = ""
        for child in self.dictionnaire_P2D_l:
            results = results + "target" + "->" + child + ";"
        return utils.drawGraph(results[:-1])

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes réduit. 
    """
    def __init__(self, df, x):
        """
        Initialise le classifieur.
        *les parametres:
        df: dataframe. 
        x: seuil de confiance.
        """
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        self.dictionnaire_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            if not isIndepFromTarget(df,attr,x):
                self.dictionnaire_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum à posteriori à partir de dictionnaire_res.
        *le paramtre attrs: le dictionnaire nom-valeur des attributs
        *le return:
        la classe 0 ou 1 estimée
        """
        dictionnaire_res = self.estimProbas(attrs)
        if dictionnaire_res[0] >= dictionnaire_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
        *le parametre attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for key in self.dictionnaire_P2D_l:
            dictionnaire_p = self.dictionnaire_P2D_l[key]
            if attrs[key] in dictionnaire_p[0]:
                P_0 *= dictionnaire_p[0][attrs[key]]
                P_1 *= dictionnaire_p[1][attrs[key]]
            else:
                return {0: 0.0, 1: 0.0}
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}
    
    def draw(self):
        """
        Construit un graphe orienté représentant naïve Bayes réduit.
        """
        results = ""
        for child in self.dictionnaire_P2D_l:
            results = results + "target" + "->" + child + ";"
        return utils.drawGraph(results[:-1])


#Question7 : évaluation des classifieurs
#Question7.2 : 
        
def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifiers à partir d'un dictionnaire dic de 
    {nom:instance de classifier} et d'un dataframe df, dans l'espace (précision,rappel). 
    *les parametres:
    dic: dictionnaire 
    df: dataframe.
    """
    precision = np.empty(len(dic))
    rappel = np.empty(len(dic))
    
    for i, nom in enumerate(dic):
         dictionnaire_stats = dic[nom].statsOnDF(df)
         precision[i] = dictionnaire_stats["Précision"]
         rappel[i] = dictionnaire_stats["Rappel"]
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(precision, rappel, marker = 'x', c = 'red') 
    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))
    plt.show()
  
    
#Question8 : Sophistication du modèle (question BONUS)
#Question8.1 : calcul des informations mutuelles

def MutualInformation(df, x, y):
    """
    Calcule l'information mutuelle entre les colonnes x et y du dataframe.
    *les parametres:
    df: le Dataframe. 
    x: nom d'une colonne du dataframe.
    y: nom d'une colonne du dataframe.
    """
    list_x = np.unique(df[x].values) # Valeurs possibles de x.
    list_y = np.unique(df[y].values) # Valeurs possibles de y.
    dictionnaire_x = {list_x[i]: i for i in range(list_x.size)} #association de chaque valeur a leur indice en list_x dans un dictionnaire.
    dictionnaire_y = {list_y[i]: i for i in range(list_y.size)} #association de chaque valeur a leur indice en list_y dans un dictionnaire.
    matrix_xy = np.zeros((list_x.size, list_y.size), dtype = int) #matrice des valeurs P(x,y)
    group = df.groupby([x, y]).groups
    for i, j in group:
        matrix_xy[dictionnaire_x[i], dictionnaire_y[j]] = len(group[(i, j)]) 
    matrix_xy = matrix_xy / matrix_xy.sum()
    matrix_x = matrix_xy.sum(1) #P(x)
    matrix_y = matrix_xy.sum(0) #P(y)
    matrix_px_py = np.dot(matrix_x.reshape((matrix_x.size, 1)),matrix_y.reshape((1, matrix_y.size))) #P(x)P(y)
    matrix_res = matrix_xy / matrix_px_py
    matrix_res[matrix_res == 0] = 1 #On evite les 0 pour des raisons de calcul
    matrix_res = np.log2(matrix_res)
    matrix_res *= matrix_xy
    return matrix_res.sum()


def ConditionalMutualInformation(df,x,y,z):
    """
    Calcule l'information mutuelle conditionnelle entre les colonnes x et y du 
    dataframe en considerant les deux comme dependantes de la colonne z.
    *les parametres:
    df: le Dataframe. 
    x: nom d'une colonne du dataframe.
    y: nom d'une colonne du dataframe.
    z: nom d'une colonne du dataframe.
    """
    #meme algo au début + l'ajoute du z
    list_x = np.unique(df[x].values) 
    list_y = np.unique(df[y].values)
    list_z = np.unique(df[z].values) 
    dictionnaire_x = {list_x[i]: i for i in range(list_x.size)} 
    dictionnaire_y = {list_y[i]: i for i in range(list_y.size)} 
    dictionnaire_z = {list_z[i]: i for i in range(list_z.size)} 
    matrix_xyz = np.zeros((list_x.size, list_y.size, list_z.size), dtype = int) #P(x,y,z)  
    group = df.groupby([x, y, z]).groups
    for i, j, k in group:
        matrix_xyz[dictionnaire_x[i], dictionnaire_y[j], dictionnaire_z[k]] = len(group[(i, j, k)]) 
    matrix_xyz = matrix_xyz / matrix_xyz.sum()
    matrix_xz = matrix_xyz.sum(1) #P(x,z)
    matrix_yz = matrix_xyz.sum(0) #P(y,z)
    matrix_z = matrix_xz.sum(0) #P(z)
    matrix_pxz_pyz = matrix_xz.reshape((list_x.size, 1, list_z.size)) * matrix_yz.reshape((1, list_y.size, list_z.size)) #P(x,z)P(y,z)
    matrix_pxz_pyz[matrix_pxz_pyz == 0] = 1
    matrix_pz_pxyz = matrix_z.reshape((1, 1, list_z.size)) * matrix_xyz #P(z)P(x, y, z)
    matrix_res = matrix_pz_pxyz / matrix_pxz_pyz
    matrix_res[matrix_res == 0] = 1
    matrix_res = np.log2(matrix_res)
    matrix_res *= matrix_xyz
    return matrix_res.sum()


##Question8.2 : calcul de la matrice des poids
    
def MeanForSymetricWeights(a):   
    """
    Calcule la moyenne des poids pour une matrice a symétrique de diagonale nulle.
    *le parametre a:
    Matrice symétrique de diagonale nulle.
    *le return : 
    la moyenne des poids
    """
    return a.sum()/(a.size - a.shape[0])

def SimplifyConditionalMutualInformationMatrix(a):
    """
    Annule toutes les valeurs plus petites que sa moyenne dans une matrice a 
    symétrique de diagonale nulle.
    *le parametre a: Matrice symétrique de diagonale nulle.      
    """
    moy = MeanForSymetricWeights(a)
    a[a < moy] = 0


#Question8.3 : Arbre (forêt) optimal entre les attributs
    
def Kruskal(df,a):
    """
    Applique l'algorithme de Kruskal au graphe dont les sommets sont les colonnes
    de df (sauf 'target') et dont la matrice d'adjacence ponderée est a.    
    *les parametres: 
    df : le Dataframe. 
    a: Matrice symétrique de diagonale nulle.
    *le return: 
    la liste des arcs , sous la forme d'une liste de triplet.
    """
    list_col = [x for x in df.keys() if x != "target"]
    list_arr = [(list_col[i], list_col[j], a[i, j]) for i in range(a.shape[0]) for j in range(i + 1, a.shape[0]) if a[i, j] != 0]
    list_arr.sort(key = lambda x: x[2], reverse = True)
    g = Graphe(list_col)
    for (u, v, poids) in list_arr:
        if g.find(u) != g.find(v):
            g.addArete(u, v, poids)
            g.union(u, v)
    return g.graphe    

class Graphe:
    """
    Structure de graphe pour l'algorithme de Kruskal. 
    """
  
    def __init__(self, sommets): 
        """
        *le parametre sommets: liste de sommets.
        """
        self.S = sommets 
        self.graphe = [] 
        self.parent = {s : s for s in self.S}
        self.taille = {s : 1 for s in self.S}
        
    def addArete(self, u, v, poids): 
        """
        Ajoute l'arete (u, v) avec poids.
        *les parametres:
        u: le nom d'un sommet.
        v: le nom d'un sommet.
        poids: poids de l'arete entre les deux sommets.
        """
        self.graphe.append((u,v,poids)) 
  
    def find(self, u): 
        """
        Trouve la racine du sommet u dans la forêt utilisée par l'algorithme de
        kruskal. Avec compression de chemin.
        *le parametre u: le nom d'un sommet.
        *le return : 
        la racine du somment u.
        """
        root = u
        #recherche de la racine
        while root != self.parent[root]:
            root = self.parent[u]
        #compression du chemin    
        while u != root:
            v = self.parent[u]
            self.parent[u] = root
            u = v
        return root            
  

    def union(self, u, v):
        """
        Union ponderé des deux arbres contenant u et v.
        *les parametres: 
        u: le nom d'un sommet.
        v: le nom d'un sommet.
        """
        u_root = self.find(u) 
        v_root = self.find(v) 
        if self.taille[u_root] < self.taille[v_root]: 
            self.parent[u_root] = v_root 
            self.taille[v_root] += self.taille[u_root] 
        else: 
            self.parent[v_root] = u_root 
            self.taille[u_root] += self.taille[v_root] 
 
#Question8.4 : Orientation des arcs entre attributs

def ConnexSets(list_arcs):
    """
    Costruit une liste des composantes connexes du graphe dont la liste d'aretes
    est list_arcs.
    *le parametre list_arcs: 
    liste de triplets de la forme (sommet1, sommet2, poids).
    *le return :
    liste des composantes connexes du graphe dont la liste d'aretes
    est list_arcs.
    """
    results = []
    for (u, v, _) in list_arcs:
        u_set = None
        v_set = None
        for s in results:
            if u in s:
                u_set = s
            if v in s:
                v_set = s
        if u_set is None and v_set is None:
            results.append({u, v})
        elif u_set is None:
            v_set.add(u)
        elif v_set is None:
            u_set.add(v)
        elif u_set != v_set:
            results.remove(u_set)
            v_set = v_set.union(u_set)
    return results

def OrientConnexSets(df, arcs, classe):
    """
    Utilise l'information mutuelle (entre chaque attribut et la classe) pour
    proposer pour chaque ensemble d'attributs connexes une racine et qui rend 
    la liste des arcs orientés.
    *les parametres:
    df: le Dataframe. 
    arcs: liste d'ensembles d'arcs connexes.
    classe: colonne de réference dans le dataframe pour le calcul de l'information mutuelle.
    *le return:
     la liste des arcs orientés  
    """
    arcs_copy = arcs.copy()
    list_sets = ConnexSets(arcs_copy)
    list_arbre = []
    for s in list_sets:
        col_max = ""
        i_max = -float("inf") #initialisation pour le max, il faut qu'il soit le plus petit possible au début.
        for col in s:
            i = MutualInformation(df, col, classe)
            if i > i_max:
                i_max = i
                col_max = col
        list_arbre += creeArbre(arcs_copy, col_max)
    return list_arbre
    
def creeArbre(arcs, racine): 
    """
    À partir d'une liste d'arcs et d'une racine, renvoie l'arbre orienté depuis
    cette racine. 
    *les parametres:
    arcs: liste d'ensembles d'arcs connexes.
    racine: nom d'un sommet.
    *le return:
    l'arbre orienté.
    """
    results = []
    file = [racine]
    while file != []:
        sommet = file.pop(0)
        arcs_copy = arcs.copy()
        for (u, v, poids) in arcs_copy:
            if sommet == u:
                results.append((u, v))
                arcs.remove((u, v, poids))
                file.append(v)
            elif sommet == v:
                results.append((v, u))
                arcs.remove((u, v, poids))
                file.append(u)
    return results 

#Question8.5 : Classifieur TAN 

class MAPTANClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle TAN
    (tree-augmented naïve Bayes).
    """
    def __init__(self, df):
        """
        Initialise le classifieur. 
        *le parametre df: le dataframe.
        """
        self.createCmis(df)
        arcs = Kruskal(df, self.cmis)
        self.liste_arcs = OrientConnexSets(df, arcs, "target")
        self.dictionnaire_P2D_l = {}
        self.dictionnaire_P3D_l = {}
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            father = self.is3D(attr)
            if father is not False:
                self.dictionnaire_P3D_l[attr] = P3D_l(df, attr, father)
            else:
                self.dictionnaire_P2D_l[attr] = P2D_l(df, attr)
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimation est faite par maximum à posteriori à partir de dictionnaire_res.
        *le parametre attrs:
        le dictionnaire nom-valeur des attributs
        *le return:
        la classe 0 ou 1 estimée
        """
        dictionnaire_res = self.estimProbas(attrs)
        if dictionnaire_res[0] >= dictionnaire_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité a posteriori P(target | attr1, ..., attrk) par
        la méthode TAN (tree-augmented naïve Bayes).
        *le parametre attrs:
        le dictionnaire nom-valeur des attributs
        *le return: 
        la valeur de la proba.
        """
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for key in self.dictionnaire_P2D_l:
            dictionnaire_p = self.dictionnaire_P2D_l[key]
            if attrs[key] in dictionnaire_p[0]:
                P_0 *= dictionnaire_p[0][attrs[key]]
                P_1 *= dictionnaire_p[1][attrs[key]]
            else:
                return {0: 0.0, 1: 0.0}
        for key in self.dictionnaire_P3D_l:
            proba = self.dictionnaire_P3D_l[key]
            P_0 *= proba.getProba(attrs[key], attrs[proba.father], 0)
            P_1 *= proba.getProba(attrs[key], attrs[proba.father], 1)
        if (P_0 + P_1) == 0 : 
            return {0: 0.0, 1: 0.0}
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}
    
    def draw(self):
        """
        Construit un graphe orienté représentant le modèle TAN.
        """
        results = ""
        for child in self.dictionnaire_P2D_l:
            results = results + "target" + "->" + child + ";"
        for child in self.dictionnaire_P3D_l:
            results = results + self.dictionnaire_P3D_l[child].father + "->" + child + ";"
            results = results + "target" + "->" + child + ";"
        return utils.drawGraph(results[:-1])
    
    def createCmis(self, df):
        """
        Crée la matrice de Conditional Mutual Information simplifiée à partir du dataframe df.
        *le parametre df:
        le dataframe. 
        """
        self.cmis = np.array([[0 if x == y else ConditionalMutualInformation(df, x, y, "target") 
                                 for x in df.keys() if x != "target"] for y in df.keys() if y != "target"])
        SimplifyConditionalMutualInformationMatrix(self.cmis)
        
    def is3D(self, attr):  
        """
        Détermine si l'attribut attr doit être représenté par une matrice 3D,
        c'est-à-dire s'il a un parent outre que "target" dans self.list_arcs.
        *le parametre attr:
        nom d'un attribut du dataframe.
        *le return: 
        matrice3D ou non.
        """
        for father, fils in self.liste_arcs:
            if fils == attr:
                return father
        return False
    
    
class P3D_l():
    """
    Classe pour le calcul des probabilités du type P(attr1 | attr2, target).
    """
    def __init__(self, df, attr1, attr2):
        """
        *les parametres:
        df: le dataframe. 
        attr1: nom d'une colonne du dataframe.
        attr2: nom d'une colonne du dataframe.
        """
        self.father = attr2
        list_x = np.unique(df[attr1].values) 
        list_y = np.unique(df[attr2].values) 
        self.dictionnaire_x = {list_x[i]: i for i in range(list_x.size)}     
        self.dictionnaire_y = {list_y[i]: i for i in range(list_y.size)} 
        self.mat = np.zeros((list_x.size, list_y.size, 2))
        group = df.groupby([attr1, attr2, 'target']).groups
        for i, j, k in group:
            self.mat[self.dictionnaire_x[i], self.dictionnaire_y[j], k] = len(group[(i, j, k)]) 
        #normalisation de la matrice
        quant = self.mat.sum(0)
        quant[quant == 0] = 1 
        self.mat = self.mat / quant.reshape((1, list_y.size, 2))
        
    def getProba(self, i, j, k):
        """
        Renvoie la valeur de P(attr1 = i | attr2 = j, target = k).
        *les parametres:
        i: valeur pour l'attribut attr1 de init.
        j: valeur pour l'attribut attr2 de init.
        k: valeur pour target.
        """
        if i in self.dictionnaire_x and j in self.dictionnaire_y:
            return self.mat[self.dictionnaire_x[i], self.dictionnaire_y[j], k]
        return 0.


    
    

