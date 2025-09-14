from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

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

# Columnas categóricas y numericas
categorical_columns = ["sex", "workclass", "education", "label"]
numeric_columns = ["age", "fnlwgt", "hours_per_week"]

indexers= [
    StringIndexer(inputCol=column, outputCol = column +"_index", handleInvalid='keep')
    for column in categorical_columns
]

# crear onehotencoder para las conlumnas con indices.
encoders = [
    OneHotEncoder(inputCol = column +"_index", outputCol = column + "_encoded")
    for column in categorical_columns
]

assembler = VectorAssembler(
    inputsCols = numeric_columns + [ column + "_encoded" for column in categorical_columns],
    outputCol = "features"
)
