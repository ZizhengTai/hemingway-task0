import breeze.linalg.{DenseVector, shuffle}
import org.apache.spark.sql.{SparkSession, Dataset}

object Main {

  def main(args: Array[String]): Unit = {
    val dataPath = args(0)
    val numClasses = args(1).toInt
    val stepSize = args(2).toDouble
    val regularizationFactor = args(3).toDouble
    val maxIterations = args(4).toInt

    val spark = SparkSession.builder
      .master("local")
      .appName("hemingway-task0")
      .getOrCreate()

    import spark.implicits._

    val data = new MnistLoader(dataPath)

    val numMachines = args(5).toInt
    val dss: Seq[Dataset[LabeledPoint]] = data.trainData
      .grouped(data.trainData.length / numMachines)
      .map(spark.createDataset(_))
      .toSeq

    val regr = new LogisticRegression(numClasses, stepSize, regularizationFactor)
    //regr.train(data.trainImages, data.trainLabels, maxIterations)
  }
}
