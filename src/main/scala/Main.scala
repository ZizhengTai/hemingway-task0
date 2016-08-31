package hemingway

import breeze.linalg.{DenseVector, DenseMatrix, argmax}
import org.apache.spark.{SparkConf, SparkContext}

object Main {
  def main(args: Array[String]): Unit = {
    val dataPath = args(0)
    val numClasses = args(1).toInt
    val stepSize = args(2).toDouble
    val regularizationFactor = args(3).toDouble
    val numIterations = args(4).toInt
    val numMachines = args(5).toInt

    val conf = new SparkConf()
      .setAppName("hemingway-task0")
      .setMaster(s"local[$numMachines]")
    implicit val sc = new SparkContext(conf)

    val data = new MnistLoader(dataPath)

    val regr = new DistributedLogisticRegression(numMachines, numClasses, stepSize, regularizationFactor)
    regr.train(data.trainingSet, numIterations)

    val actual = data.testLabels
    val pred = regr.predict(data.testImages map (DenseVector(_)))
    val accuracy = (actual, pred).zipped.map((a, b) => if (a == b) 1 else 0).sum.toDouble / actual.length

    println("Test accuracy: " + accuracy)
  }
}
