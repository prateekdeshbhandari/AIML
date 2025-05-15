import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

// Initialize Spark session
val spark = SparkSession.builder
  .appName("KMeansClusteringExample")
  .getOrCreate()

import spark.implicits._

// Sample data: features represent [age, income, spending_score]
val data = Seq(
  (25, 50000, 60),
  (30, 60000, 70),
  (35, 75000, 80),
  (20, 30000, 50),
  (40, 80000, 90),
  (45, 90000, 85),
  (28, 55000, 65),
  (32, 65000, 75)
)

// Create DataFrame
val df = data.toDF("age", "income", "spending_score")

// Assemble features into a single vector column
val assembler = new VectorAssembler()
  .setInputCols(Array("age", "income", "spending_score"))
  .setOutputCol("features")

val assembledDF = assembler.transform(df)

// Scale the features
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaled_features")
  .setWithStd(true)
  .setWithMean(true)

val scalerModel = scaler.fit(assembledDF)
val scaledDF = scalerModel.transform(assembledDF)

// Initialize and train KMeans model
val kmeans = new KMeans()
  .setFeaturesCol("scaled_features")
  .setPredictionCol("cluster")
  .setK(3)
  .setSeed(42)

val model = kmeans.fit(scaledDF)

// Make predictions
val predictions = model.transform(scaledDF)

// Evaluate clustering using Silhouette score
val evaluator = new ClusteringEvaluator()
  .setPredictionCol("cluster")
  .setFeaturesCol("scaled_features")
  .setMetricName("silhouette")

val silhouetteScore = evaluator.evaluate(predictions)

// Show results
println(s"Silhouette Score: $silhouetteScore")
predictions.select("age", "income", "spending_score", "cluster").show()

// Stop Spark session
spark.stop()
