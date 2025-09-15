from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
print("=== DATOS ORIGINALES ===")
df.show(5)
print(f"Número total de registros: {df.count()}")

# Columnas categóricas y numericas
categorical_columns = ["sex", "workclass", "education"]
numeric_columns = ["age", "fnlwgt", "hours_per_week"]

indexers= [
    StringIndexer(inputCol=column, outputCol = column +"_index", handleInvalid='keep')
    for column in categorical_columns
]

label_indexer = StringIndexer(inputCol="label", outputCol="label_index", handleInvalid='error')


# crear onehotencoder para las conlumnas con indices.
encoders = [
    OneHotEncoder(inputCol = column +"_index", outputCol = column + "_encoded")
    for column in categorical_columns
]

assembler = VectorAssembler(
    inputCols = numeric_columns + [ column + "_encoded" for column in categorical_columns],
    outputCol = "features"
)

lr = LogisticRegression(featuresCol='features', labelCol='label_index', maxIter=100)


## Crear pipeline
pipeline = Pipeline(stages = [label_indexer] + indexers + encoders + [assembler, lr])

## Entreanamiento del modelo
model = pipeline.fit(df)
## Predicciones
df_transformed = model.transform(df)

## Mostrar predicciones
print("\n=== MAPEO DE ETIQUETAS ===")
print("Label original -> Label indexado:")
print("  0 (<=50K) -> 0.0")
print("  1 (>50K)  -> 1.0")

df_transformed.select(
    'features',
    'label', 
    'label_index',
    'prediction',
    'probability'
).show(20, truncate=False)




evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction",
    metricName="accuracy"
)


auc = evaluator.evaluate(df_transformed)
print(f"\n=== MÉTRICAS DEL MODELO ===")
print(f"Area bajo la curva  (AUC): {auc:.4f}")

lr_model = model.stages[-1]  # El último stage es el modelo LR
print(f"\n=== INFORMACION DEL MODELO ===")
print(f"Coeficientes: {lr_model.coefficientMatrix}")
print(f"Intercepto: {lr_model.intercept}")
print(f"Número de iteraciones: {lr_model.summary.totalIterations}")

# 13. Contar predicciones por clase
print("\n=== DISTRIBUCIÓN DE PREDICCIONES ===")
df_transformed.groupBy("prediction").count().show()
df_transformed.groupBy("label").count().show()


