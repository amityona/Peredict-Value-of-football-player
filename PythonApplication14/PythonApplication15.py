import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import scipy.stats as stats
import re
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import sklearn.metrics as metrics
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
from bs4 import BeautifulSoup
import patsy
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model ,metrics
from yellowbrick.regressor import ResidualsPlot
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.express as px

DataOfPlayerArr = []

i = 0
while i < 600:
    urlSoFifa = "https://sofifa.com/?&showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=wi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=bo&showCol%5B%5D=bp&showCol%5B%5D=gu&showCol%5B%5D=vl&showCol%5B%5D=wg&showCol%5B%5D=rc&showCol%5B%5D=ta&showCol%5B%5D=cr&showCol%5B%5D=fi&showCol%5B%5D=he&showCol%5B%5D=sh&showCol%5B%5D=vo&showCol%5B%5D=ts&showCol%5B%5D=dr&showCol%5B%5D=cu&showCol%5B%5D=fr&showCol%5B%5D=lo&showCol%5B%5D=bl&showCol%5B%5D=to&showCol%5B%5D=ac&showCol%5B%5D=sp&showCol%5B%5D=ag&showCol%5B%5D=re&showCol%5B%5D=ba&showCol%5B%5D=tp&showCol%5B%5D=so&showCol%5B%5D=ju&showCol%5B%5D=st&showCol%5B%5D=sr&showCol%5B%5D=ln&showCol%5B%5D=te&showCol%5B%5D=ar&showCol%5B%5D=in&showCol%5B%5D=po&showCol%5B%5D=vi&showCol%5B%5D=pe&showCol%5B%5D=cm&showCol%5B%5D=td&showCol%5B%5D=ma&showCol%5B%5D=sa&showCol%5B%5D=sl&showCol%5B%5D=tg&showCol%5B%5D=gd&showCol%5B%5D=gh&showCol%5B%5D=gc&showCol%5B%5D=gp&showCol%5B%5D=gr&showCol%5B%5D=tt&showCol%5B%5D=bs&showCol%5B%5D=ir&showCol%5B%5D=pac&showCol%5B%5D=sho&showCol%5B%5D=pas&showCol%5B%5D=dri&showCol%5B%5D=def&showCol%5B%5D=phy&offset="+str(i)
    response = requests.get(urlSoFifa)
    pageSoFifa = response.text
    bPage = BeautifulSoup(pageSoFifa,"html.parser")
    trFoundHtml = bPage.find_all("tr")
    for dataCatch in trFoundHtml:
        tdFound = dataCatch.find_all('td')
        tdToString = str(tdFound)
        patern = re.compile('<.*?>')
        afterEdit = re.sub(patern,'',tdToString)
        DataOfPlayerArr.append(afterEdit)
        
    print(i)
    i += 60
newAnswerToFix = pd.DataFrame(DataOfPlayerArr)
print("Number of All Rows")
print(len(newAnswerToFix));
print("Number of Rows Without Duplicated")
newAnswerToFix.drop_duplicates(subset=None, keep="first", inplace=True)
print(len(newAnswerToFix))
ChoseCol = [
 'Name',
 'Age',
 'Overall',
 'Potential',
 'Team',
 'Height',
 'Weight',
 'Foot',
 'BestOverall',
 'Position',
 'Growth',
 'Value',
 'Wage',
 'ReleaseClause',
 'Attacking',
 'Cross',
 'Finish',
 'HeadingAccuraci',
 'ShortPassing',
 'Volley',
 'Skill',
 'Dribbling',
 'Curve',
 'FKAccuracy',
 'PassLong',
 'BallControl',
 'Movement',
 'Accelerat',
 'SprintSpeed',
 'Agiliti',
 'Reaction',
 'Balance',
 'Power',
 'PowerShoT',
 'Jumping',
 'Stamina',
 'Strength',
 'LongShots',
 'Mentali',
 'Aggress',
 'Intercept',
 'Positioning',
 'Vision',
 'Penalti',
 'Composure',
 'Defending',
 'Marking',
 'StandingTackle',
 'SlidingTackle',
 'Goalkeeping',
 'GKDiving',
 'GKHandling',
 'GKKicking',
 'GKPositioning',
 'GKReflexes',
 'TotalStats',
 'BaseStats',
 'IntReputation',
 'PAC',
 'SHO',
 'PAS',
 'DRI',
 'DEF',
 'PHY']
newAnswerToFix = newAnswerToFix[0].str.split(',',expand=True)
newAnswerToFix[1] = newAnswerToFix[1].str.replace("\n ","")
posFix = ["GK","RB","RCB","RS","LCB","LB","RWB","RDM","ST","RCM","LWB","RM","LDM","LCM","LS","LM","LAM","RW","RF","CAM","RAM","CF","LF","LW","CM","CDM","CB"]
for pos in posFix:
    newAnswerToFix[1]=newAnswerToFix[1].str.replace(pos,"")
    newAnswerToFix[1]=newAnswerToFix[1].str.lstrip()
newAnswerToFix[5] = newAnswerToFix[5].str.replace("\n\n\n\n","")
newAnswerToFix[5] = newAnswerToFix[5].str.split("\n").str[0]
newAnswerToFix[5] = newAnswerToFix[5].str.strip()
newAnswerToFix[6] = newAnswerToFix[6].replace(r'Loan', np.nan, regex=True)
newAnswerToFix = newAnswerToFix.iloc[1:]
AnswerCurrent = newAnswerToFix[newAnswerToFix[6].notna()]
AnswerCurrent.drop(AnswerCurrent.columns[[0,6,66]],axis=1, inplace=True)
nextFixDa = newAnswerToFix[6].isnull()
result_loan = newAnswerToFix.loc[nextFixDa].shift(-1, axis=1)
result_loan.drop_duplicates(subset=[6], keep="first", inplace=True)
result_loan.drop(result_loan.columns[[5,6,66]],axis=1, inplace=True)
print(AnswerCurrent.columns)
AnswerCurrent.columns = ChoseCol
result_loan.columns = ChoseCol
resuToSave = pd.concat([AnswerCurrent,result_loan]).reset_index()
resuToSave.drop_duplicates(subset=None, keep="first",inplace=True)
resuToSave = resuToSave[~resuToSave["Height"].str.contains("~")]
resuToSave = resuToSave[~resuToSave["Age"].str.contains("\n")]
resuToSave = resuToSave[~resuToSave["Height"].str.contains("'")]
resuToSave["Height"] = resuToSave["Height"].str[:-1]
resuToSave["Height"] = resuToSave["Height"].str.replace("c","")
resuToSave["Value"] = resuToSave["Value"].str.replace("€","")
resuToSave["ReleaseClause"] = resuToSave["ReleaseClause"].str.replace("€","")
print(resuToSave)
resuToSave["ReleaseClause"] = resuToSave["ReleaseClause"].str.replace("M","")
resuToSave["Value"] = resuToSave["Value"].str.replace("M","")
for curPos in posFix:
    resuToSave["Wage"] = resuToSave["Wage"].str.replace(curPos,"0K")
    resuToSave["ReleaseClause"] = resuToSave["ReleaseClause"].str.replace(curPos,"0K")
resuToSave["Wage"] = resuToSave["Wage"].str.replace("€","")
resuToSave.loc[resuToSave["Value"].str.contains("K"),"Value"]=resuToSave["Value"].str.split("K").str[0].astype(float)/1000
resuToSave["Value"] = round(resuToSave["Value"].astype(float),2)
resuToSave = resuToSave.loc[resuToSave["Value"]>0]
resuToSave.loc[resuToSave["Wage"].str.contains("K"),"Wage"]=resuToSave["Wage"].str.split("K").str[0].astype(float)*1000
resuToSave["Wage"] = round(resuToSave["Wage"].astype(float)/1000000,5)
resuToSave.loc[resuToSave["ReleaseClause"].str.contains("K"),"ReleaseClause"]=resuToSave["ReleaseClause"].str.split("K").str[0].astype(float)/1000
resuToSave["ReleaseClause"] = round(resuToSave["ReleaseClause"].astype(float),2)
resuToSave = resuToSave.loc[resuToSave["Composure"]!=" "]
resuToSave = resuToSave[resuToSave["BaseStats"]!=' ']
resuToSave["IntReputation"]=resuToSave["IntReputation"].str[:-1]
intCol=[
 'Age',
 'Overall',
 'Potential',
 'BestOverall',
 'Growth',
 'Attacking',
 'Cross',
 'Finish',
 'HeadingAccuraci',
 'ShortPassing',
 'Volley',
 'Skill',
 'Dribbling',
 'Curve',
 'FKAccuracy',
 'PassLong',
 'BallControl',
 'Movement',
 'Accelerat',
 'SprintSpeed',
 'Agiliti',
 'Reaction',
 'Balance',
 'Power',
 'PowerShoT',
 'Jumping',
 'Stamina',
 'Strength',
 'LongShots',
 'Mentali',
 'Aggress',
 'Intercept',
 'Positioning',
 'Vision',
 'Penalti',
 'Composure',
 'Defending',
 'Marking',
 'StandingTackle',
 'SlidingTackle',
 'Goalkeeping',
 'GKDiving',
 'GKHandling',
 'GKKicking',
 'GKPositioning',
 'GKReflexes',
 'TotalStats',
 'BaseStats',
 'IntReputation',
 'PAC',
 'SHO',
 'PAS',
 'DRI',
 'DEF']


for currentColFix in intCol:
    resuToSave[currentColFix] = resuToSave[currentColFix].astype("int")



resuToSave.to_csv('FixData.csv')  

############################################

FixDataPlayer = pd.read_csv('FixData.csv')

print(FixDataPlayer)
print(" \n \n             Top 10 Most Valuable")


FixDataPlayer.hist(bins=15,color='steelblue',edgecolor='red',linewidth=1.0,
                    xlabelsize=8,ylabelsize=8,grid=False)
plt.tight_layout(rect=(0,0,1.2,1.2))
print(FixDataPlayer.nlargest(10,columns="Value")[["Name","Overall","Team","Position","Value"]])
print(FixDataPlayer.nlargest(10,columns="Overall")[["Name","Value","Team","Position","Overall"]])

print("Top 10 Most Value Position")

print(pd.DataFrame(FixDataPlayer.groupby("Team").Value.mean().sort_values(ascending=False)).head(10))


print("Top 10 Most Overall Team")
print(pd.DataFrame(FixDataPlayer.groupby("Team").Overall.mean().sort_values(ascending=False)).head(10))

club = FixDataPlayer.groupby('Team')['Value'].mean().reset_index().sort_values('Value', ascending=True).tail(10)
fig = px.bar(club, x="Value", y="Team", orientation='h')
fig.show()

print('''"FixDataPlayer[["Age","Value","Overall","TotalStats","Weight"]].describe())"''')
print(FixDataPlayer[["Age","Value","Overall","TotalStats","Weight"]].describe())


plt.figure(figsize=(10,7))
sns.boxplot(x='Age', data=FixDataPlayer, width=2)
plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(x='Value', data=FixDataPlayer, width=2)
plt.show()

plt.figure(figsize=(10,7))
sns.boxenplot(x='TotalStats', data=FixDataPlayer, width=2)
plt.show()
plt.figure(figsize=(10,7))
sns.boxenplot(x='Overall', data=FixDataPlayer, width=2)
plt.show()


plt.figure(figsize=(10,6))
sns.regplot(x="Value",y="Overall",data=FixDataPlayer)
plt.show()

FixDataPlayer.hist(bins=15, figsize=(15,10));

x = FixDataPlayer['Age']
plt.figure(figsize=(10,8))
ax = sns.countplot(x,color='#9b59b6')
ax.set_xlabel(xlabel = 'Age', fontsize = 12)
ax.set_title(label = 'Distribution - Players age', fontsize = 14)
plt.show()

x = FixDataPlayer['Position']
plt.figure(figsize=(10,8))
ax = sns.countplot(x,color='#9b59b6')
ax.set_xlabel(xlabel = 'Position', fontsize = 12)
ax.set_title(label = 'Distribution - Players Position', fontsize = 14)
plt.show()

overall = pd.DataFrame(FixDataPlayer.groupby(["Age"])['Overall'].mean())
potential = pd.DataFrame(FixDataPlayer.groupby(["Age"])['Potential'].mean())

merged = pd.merge(overall, potential, on='Age', how='inner')
merged['Age']= merged.index

fig, ax = plt.subplots(figsize=(8,6))

merged.reset_index(drop = True, inplace = True)

plt.plot('Age', 'Overall', data=merged, marker='.', color='#00ffff', lw=1, label ="Overall" )
plt.plot('Age', 'Potential', data=merged, marker='+', color='#9b59b6', lw=1, label = "Potential")
plt.xlabel('Overall Rating')
plt.ylabel('Average Growth Potential by Age')
plt.legend();


FixDataPlayer.hist(bins=15,color='steelblue',edgecolor='red',linewidth=1.0,
                    xlabelsize=8,ylabelsize=8,grid=False)
plt.tight_layout(rect=(0,0,1.2,1.2))

FixDataPlayer = FixDataPlayer[['Value','Overall','Age','Name','Cross','Jumping','BestOverall',
                           'Position','Attacking','Potential','Finish',
                           'HeadingAccuraci','ShortPassing','Volley','Skill','Stamina','Curve',
                           'PassLong','BallControl','Movement','Accelerat','SprintSpeed','Agiliti',
                           'Reaction','Balance','Power','PowerShoT','Team','Dribbling','Strength','LongShots',
                           'StandingTackle','Positioning','Defending','Marking','Mentali','SlidingTackle','Goalkeeping',
                           'GKReflexes','TotalStats','BaseStats',
                           'PAC','SHO','PAS','DRI','DEF','PHY',
                           'Growth','Height','IntReputation','ReleaseClause','Weight']]

print("****************** corr player List *******************")
corr_mat = FixDataPlayer.corr()
print(corr_mat)


print("****************** @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ *******************")
print(pd.DataFrame(corr_mat["Value"]).sort_values("Value", ascending=False).head(10))
print(pd.DataFrame(corr_mat["Value"]).sort_values("Value", ascending=True).head(7))



FixDataPlayer["ValueLog"] = np.log(FixDataPlayer["Value"])

ColPlayerListAttritube= FixDataPlayer[['ValueLog','Overall','TotalStats','ReleaseClause','BestOverall','IntReputation']]
print(ColPlayerListAttritube.corr())
plt.figure(1, figsize=(20, 8))
sns.set(style="whitegrid")
g=sns.pairplot(ColPlayerListAttritube, height=1.7, aspect=1.9)
plt.yticks(rotation=90); 
plt.show()


ColPlayerListAttritube= FixDataPlayer[['ValueLog','Overall','TotalStats','ReleaseClause','BestOverall','Age','IntReputation']]
plt.figure(1, figsize=(18, 7))
sns.set(style="whitegrid")
sns.heatmap(ColPlayerListAttritube.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.yticks(rotation=0); 
plt.show()




os.system('cls')
print("****************** @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ *******************")
print("****************** @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ *******************")
lr = linear_model.LinearRegression() 
X = FixDataPlayer[['Overall','Age','BestOverall','TotalStats','ReleaseClause','IntReputation']] 
y = FixDataPlayer['Value'] 
sns.heatmap(FixDataPlayer.corr(), cmap='RdBu',center=0)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
model_answer = lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
y_pred_train = lr.predict(X_train) 
print("Coef::",lr.coef_)
print("Intercept:",lr.intercept_)
Xs= FixDataPlayer[['Overall','Age','BestOverall','TotalStats','ReleaseClause','IntReputation']]
ys= FixDataPlayer[['Value']] 
player_model = sm.OLS(ys, Xs, data=FixDataPlayer) 
results = player_model.fit()
print(results.summary())
y_actual=y_test
print('Train Score: ', lr.score(X_train, y_train))  
print('Test Score: ', lr.score(X_test, y_test))  
plt.figure(figsize=(12,8),dpi=150),
plt.scatter(y_pred_train,y_train)
plt.title("Regression Fit")
plt.show()
scores = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(scores.mean() * 100))
print("Standard: {:.2f} %".format(scores.std() * 100))
dfAnswer = pd.DataFrame({'Actual': y_test , 'Predicted': y_pred})
print(dfAnswer)
df1 = dfAnswer.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
plt.show()


plt.figure(figsize=(14,8),dpi=130),


plt.scatter(y_pred,y_test)
plt.title("Test Value  vs Predicted Value 2 ")
plt.xlabel("Predict Value ")
plt.ylabel("Actual Value ")
plt.show()
print("Y Pred is : \n !!!!")
print(y_pred)


def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(y_test, y_pred, color='red')
    plt.plot(range(my_range), range(my_range), 'o')
    plt.title("Graph1")
    plt.show()
    return


def plotGraph2(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title("Graph2")
    plt.show()
    return

plotGraph(y_test,y_pred,lr)
plotGraph2(y_test,y_pred,lr)


predicted = cross_val_predict(lr, X_test, y_test, cv=10)

fig, ax = plt.subplots()
ax.scatter(y_test, predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



lasso_model = Lasso()
visualizer = ResidualsPlot(lasso_model) 
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
visualizer.show("Residual_lasso.jpg")
print(f'Lasso test data R^2: {lasso_model.score(X_test, y_test):.4f}')


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df.plot(kind='line',figsize=(18,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

model = Ridge()
visualizer = ResidualsPlot(model)
visualizer = ResidualsPlot(model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

data = FixDataPlayer.tail(100)
data = data.head(50)

X12 = data[['Overall','Age','BestOverall','TotalStats','ReleaseClause','IntReputation']] 
ynew = data['Value'] 
y_pred2 = lr.predict(X12)
np.set_printoptions(precision=2)
print(y_pred2)
df = pd.DataFrame({'Name':data.Name,'Actual Value': ynew ,'Predicted Value': y_pred2 , 'Age':data.Age, 'Positioning':data.Positioning}).head(40)
print(df)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print("End of Regresi")
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


X = FixDataPlayer[['Value','Age','TotalStats','IntReputation','BaseStats','BallControl','Positioning']] 
y = FixDataPlayer['Overall'] 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
from sklearn.neighbors import KNeighborsRegressor
k_range = range(1,4)
k_range = [1,3,5,7,9]
scores = {}
scores_list = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    plt.scatter(y_pred,y_test)
    plt.title("Test Value  vs Predicted Value KnN ")
    plt.xlabel("Predict Value ")
    plt.ylabel("Actual Value ")
    plt.show()
    print("now")
    scores_list.append(knn.score(X_test, y_test))
    print(knn.score(X_test, y_test))


plt.plot(k_range,scores_list)
plt.xlabel("Value Of K")
plt.ylabel("Test Accuracy")
plt.show()

best_k= ['k_range' , 'scores_list']
list_of_tuples = list(zip(k_range, scores_list))
df_best_k = pd.DataFrame(list_of_tuples, columns = ['k', 'result']) 
print(df_best_k)


df_best_k.plot(kind="bar",x='k',y='result', color='red')
plt.xlabel('k')
plt.ylabel('result')


knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train,y_train)

data = FixDataPlayer.tail(100)
data = data.head(50)
X12 = data[['Value','Age','TotalStats','IntReputation','BaseStats','BallControl','Positioning']] 
ynew = data['Overall'] 
y_pred2 = knn.predict(X12)

df = pd.DataFrame({'Name':data.Name,'Actual Overall': ynew, 'Predicted Overall': y_pred2 , 'Age':data.Age,'Positioning':data.Positioning}).head(40)
print(df)