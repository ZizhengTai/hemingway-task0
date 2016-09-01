package hemingway

import breeze.linalg._
import breeze.plot._
import org.apache.spark.{SparkConf, SparkContext}
import org.jfree.chart.axis.NumberTickUnit

object Main {
  def main(args: Array[String]): Unit = {
    val dataDir = args(0)
    val numClasses = args(1).toInt
    val stepSize = args(2).toDouble
    val regularizationFactor = args(3).toDouble
    val numIterations = args(4).toInt
    val numMachines = args(5).toInt
    val outputDir = args(6)

    val conf = new SparkConf()
      .setAppName("hemingway-task0")
      .setMaster(s"local[$numMachines]")
    implicit val sc = new SparkContext(conf)

    val data = new MnistLoader(dataDir)

    val regr = new DistributedLogisticRegression(
      numMachines,
      numClasses,
      data.trainImages(0).length,
      stepSize,
      regularizationFactor)
    regr.train(data.trainingSet, numIterations)

    val actual = data.testLabels
    val pred = regr.predict(data.testImages map (DenseVector(_)))
    val accuracy = (actual, pred).zipped.map((a, b) => if (a == b) 1 else 0).sum.toDouble / actual.length

    println("Test accuracy: " + accuracy)

    val iters = linspace(0, regr.iterationInfo.length, regr.iterationInfo.length)
    val losses = regr.iterationInfo map (_.loss)
    val iterTime = regr.iterationInfo map (_.iterTime.toMillis.toDouble)
    val totalTime = regr.iterationInfo map (_.totalTime.toMillis.toDouble)

    val f = Figure()
    f.rows = 3
    f.cols = 1
    f.width = 600
    f.height = 800

    // Training loss against iterations
    val p1 = f.subplot(0)
    p1.yaxis.setTickUnit(new NumberTickUnit((losses.max - losses.min) / 5))
    p1.xlabel = "Iterations"
    p1.ylabel = "Training loss"
    p1 += plot(iters, losses)

    // Time taken per iteration against iterations
    val p2 = f.subplot(1)
    p2.yaxis.setAutoRangeIncludesZero(true)
    p2.xlabel = "Iterations"
    p2.ylabel = "Time per iteration (ms)"
    p2 += plot(iters, iterTime)

    // Training loss against total time
    val p3 = f.subplot(2)
    p3.yaxis.setTickUnit(new NumberTickUnit((losses.max - losses.min) / 5))
    p3.xlabel = "Total time (ms)"
    p3.ylabel = "Training loss"
    p3 += plot(totalTime, losses)

    f.saveas(outputDir + "/out.png")
  }
}
