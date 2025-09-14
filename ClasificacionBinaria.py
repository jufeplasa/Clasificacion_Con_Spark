import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

conf = SparkConf().setMaster("local").setAppName("Clasificacion_Binaria")
sc = SparkContext(conf = conf)

lines = sc.textFile("file:///SparkCourse/fakefriends.csv")
rdd = lines.map(parseLine)