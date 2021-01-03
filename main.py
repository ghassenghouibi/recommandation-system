#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Recommandation system


# In[10]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from utils import get_mat_sparsity
from pyspark.sql.functions import explode
from pyspark.sql.functions import col


# In[5]:


sc = pyspark.SparkContext('local[*]')


# In[6]:


ratingsPath = 'datasets/ratings.csv'
moviesPath  = 'datasets/movies.csv'


# In[210]:


spark = SparkSession.builder.appName('Recommendations').getOrCreate()
movies = spark.read.csv(moviesPath,header=True)
movies.printSchema()
ratings = spark.read.csv(ratingsPath,header=True)
ratings.printSchema()
print(ratings.count())
ratings=ratings.drop('timestamp')
movies.show(5)
ratings.show(5)


# In[11]:


movie_ratings = ratings.join(movies, ['movieId'], 'left')
movie_ratings = movie_ratings.withColumn("userId", ratings["userId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("movieId", ratings["movieId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("rating", ratings["rating"].cast('float'))
movie_ratings.show(5)
#sparsity
get_mat_sparsity(movie_ratings)


# In[12]:


#split train and test
(train, test) = movie_ratings.randomSplit([0.8, 0.2], seed = 2020)
train.printSchema()


# In[13]:


# ALS model
als = ALS(
         userCol="userId", 
         itemCol="movieId",
         ratingCol="rating", 
         nonnegative = True, 
         implicitPrefs = False,
         coldStartStrategy="drop"
)


# In[18]:


param_grid = ParamGridBuilder()             .addGrid(als.rank, [10, 25, 50, 100])             .addGrid(als.regParam, [.01, .05, .1, .15])             .build()
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
model = cv.fit(train)
best_model = model.bestModel

print("**Best Model**")
print("  Rank:", best_model._java_obj.parent().getRank())
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
print("  RegParam:", best_model._java_obj.parent().getRegParam())


# In[19]:


als = ALS(
        maxIter=10,
        regParam=0.15,
        rank=50,
        userCol='userId',
        itemCol='movieId',
        ratingCol='rating',
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy='drop'
)
model = als.fit(train)


# In[20]:


test_predictions = model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)


# In[21]:


recommendations = model.recommendForAllUsers(5)
recommendations.show()


# In[43]:


nrecommendations = recommendations    .withColumn("rec_exp", explode("recommendations"))    .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))


# In[44]:


nrecommendations.limit(10).show()


# In[46]:


user=nrecommendations.join(movies, on='movieId').filter('userId = 77')


# In[54]:


userRated=movie_ratings.filter('userId = 77').sort('rating',ascending=False).limit(5)


# In[241]:


picUsersRecommendation=[]
userRated=[]
usersIds=[7,77,99]
for i in range(0,3):
    user=nrecommendations.join(movies, on='movieId').filter('userId = '+ str(usersIds[i]))
    userRating=movie_ratings.filter('userId = '+ str(usersIds[i])).sort('rating',ascending=False).limit(5)
    picUsersRecommendation.append(user.toPandas())
    userRated.append(userRating.toPandas())


# In[48]:


result=nrecommendations.join(movies, on='movieId')


# In[28]:


result.repartition(1).write.csv('output/ouput.csv',header=True)


# In[242]:


import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

font = {'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
for i in range(0,3):
    fig=pd.DataFrame(picUsersRecommendation[i])
    x_labels = fig['genres'].values
    y_labels = fig['rating'].values
    df = pd.DataFrame({'genres': x_labels,'rating': y_labels})
    df.plot.bar(x='genres',y='rating',subplots=True,color='red',figsize=(5,5),rot=45)

    fig=pd.DataFrame(userRated[i])
    x_labels = fig['genres'].values
    y_labels = fig['rating'].values
    df = pd.DataFrame({'genres': x_labels,'rating': y_labels})
    df.plot.bar(x='genres',y='rating',subplots=True,color='green',figsize=(5,5),rot=45)


# In[231]:


bestGender=nrecommendations.join(movies, on='movieId').limit(10)
from pyspark.sql.functions import col,when,count

bestGender.show()
BestAndWorst=bestGender.groupBy("genres").agg(
    count(when(col("rating") >= 3.0, True)).alias('best'),
    count(when(col("rating") <= 2.9, True)).alias('mean'))


BestAndWorst.show()
BestAndWorst=pd.DataFrame(BestAndWorst.toPandas())

x_labels = BestAndWorst['genres'].values
y_labels = BestAndWorst['best'].values

ax = plt.subplot(111)
ax.bar(x_labels, y_labels, width=0.2, color='b', align='center')

plt.show()


# In[239]:


bestFilms=nrecommendations.join(movies, on='movieId')

bestFilms=bestFilms.groupBy("title").agg(
    count(when(col("rating") >= 3.0, True)).alias('best'),
    count(when(col("rating") <= 2.9, True)).alias('mean')).sort('best',ascending=False)


bestFilms.show()
bestFilms=pd.DataFrame(bestFilms.toPandas())

x_labels = bestFilms['title'].values
print(x_labels[:10])


# In[ ]:




