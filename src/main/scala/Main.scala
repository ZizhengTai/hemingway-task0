package hemingway

import breeze.linalg.{DenseVector, DenseMatrix, argmax}
import org.apache.spark.sql.SparkSession

object Main {

  def main(args: Array[String]): Unit = {
    val dataPath = args(0)
    val numClasses = args(1).toInt
    val stepSize = args(2).toDouble
    val regularizationFactor = args(3).toDouble
    val maxIterations = args(4).toInt
    val numMachines = args(5).toInt

    val spark = SparkSession.builder
      .master("local")
      .appName("hemingway-task0")
      .getOrCreate()

    import spark.implicits._

    val data = new MnistLoader(dataPath)
    val trainingSets = data.trainingSet.split(numMachines).toDS

    val avgParams = new DenseMatrix(
      numClasses,
      data.trainImages(0).length,
      trainingSets map { labeledDataset =>
        val x = labeledDataset.data map (lp => DenseVector(lp.features))
        val y = labeledDataset.data map (_.label)

        val regr = new LogisticRegression(numClasses, stepSize, regularizationFactor)
        regr.train(x, y, maxIterations)

        regr.params.data
      } reduce { (p1, p2) =>
        (DenseVector(p1) + DenseVector(p2)).data
      }
    ) / numMachines.toDouble

    def predict(x: DenseVector[Double]): Int = argmax(avgParams * x)
    val actual = data.testLabels
    val pred = data.testImages map (x => predict(DenseVector(x)))
    val accuracy = (actual, pred).zipped.map((a, b) => if (a == b) 1 else 0).sum.toDouble / actual.length

    println("Test accuracy: " + accuracy)
  }
}
