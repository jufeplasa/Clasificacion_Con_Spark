import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from math import sqrt

spark = SparkSession.builder \
    .master("local") \
    .appName("Clasificacion_Binaria") \
    .getOrCreate()

# Leer directamente como DataFrame (maneja encabezados automáticamente)
df = spark.read.csv("../adult_income_sample.csv", 
                   header=True, 
                   inferSchema=True)

# Mostrar esquema y datos
df.printSchema()
df.show(5)

# Convertir a RDD si lo necesitas
rdd = df.rdd.map(lambda row: tuple(row))
print(rdd.take(5))