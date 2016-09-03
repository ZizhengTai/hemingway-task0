package hemingway

import scala.util.Random
import breeze.linalg._
import breeze.plot._
import org.apache.spark.{SparkConf, SparkContext}
import org.jfree.chart.axis.NumberTickUnit
import org.jfree.ui.{HorizontalAlignment, RectangleEdge, VerticalAlignment}

object Main {
  def main(args: Array[String]): Unit = {
    // Extract arguments
    val dataDir = args(0)
    val numClasses = args(1).toInt
    val stepSize = args(2).toDouble
    val regularizationFactor = args(3).toDouble
    val numIterations = args(4).toInt
    val numMachines = args(5).toInt
    val outputDir = args(6)

    // Create Spark context
    val conf = new SparkConf()
      .setAppName("hemingway-task0")
      .setMaster(s"local[$numMachines]")
    implicit val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    // Warm up Spark
    warmUp(numMachines)

    // Load MNIST dataset
    val data = new MnistLoader(dataDir)

    // Regression for each number of machines
    val results = Seq(1, 2, 4, 8) map { m =>
      val regr = new DistributedLogisticRegression(
        numMachines = m,
        numClasses = numClasses,
        numFeatures = data.trainImages(0).length,
        stepSize = stepSize,
        regularizationFactor = regularizationFactor)

      regr.train(data.trainingSet, numIterations)

      regr
    }

    // Plot results
    val fs = Seq.fill(3)(Figure())
    for (f <- fs) {
      f.width = 1920
      f.height = 1080
    }

    val ps = fs map (_.subplot(0))
    ps foreach (_.legend = true)

    // Training loss against iterations
    val allLosses = results flatMap (_.iterationInfo map (_.loss))
    val lossDiff = allLosses.max - allLosses.min
    //ps(0).yaxis.setTickUnit(new NumberTickUnit(lossDiff / 10))
    ps(0).yaxis.setAutoTickUnitSelection(true)
    ps(0).xlabel = "Iterations"
    ps(0).ylabel = "Training loss"

    // Time taken per iteration against iterations
    ps(1).yaxis.setAutoRangeIncludesZero(true)
    ps(1).xlabel = "Iterations"
    ps(1).ylabel = "Time per iteration (ms)"

    // Training loss against total time
    ps(2).yaxis.setTickUnit(new NumberTickUnit(lossDiff / 5))
    ps(2).xlabel = "Total time (ms)"
    ps(2).ylabel = "Training loss"

    for ((r, i) <- results zip Seq(1, 2, 4, 8)) {
      val iters = r.iterationInfo.indices map (1.0 + _)
      val losses = r.iterationInfo map (_.loss)
      val iterTime = r.iterationInfo map (_.iterTime.toMillis.toDouble)
      val totalTime = r.iterationInfo map (_.totalTime.toMillis.toDouble)

      ps(0) += plot(iters, losses, name = i.toString)
      ps(1) += plot(iters, iterTime, name = i.toString)
      ps(2) += plot(totalTime, losses, name = i.toString)
    }

    // Save all plots
    fs(0).saveas(outputDir + "/loss-iters.png")
    fs(1).saveas(outputDir + "/time-iters.png")
    fs(2).saveas(outputDir + "/loss-time.png")

    /*
    val actual = data.testLabels
    val pred = regr.predict(data.testImages map (DenseVector(_)))
    val accuracy = (actual, pred).zipped.map((a, b) => if (a == b) 1 else 0).sum.toDouble / actual.length

    println("Test accuracy: " + accuracy)
    */
  }

  /** Warms up Spark.
    *
    *  @param numMachines number of machines to warm up
    *  @param sc Spark context
    */
  def warmUp(numMachines: Int)(implicit sc: SparkContext): Unit = {
    val numDatapoints = 10000
    val numClasses = 10
    val numFeatures = 1000
    val stepSize = 1e-3
    val regularizationFactor = 1e-5
    val numIterations = 3

    val data = LabeledDataset(
      labels = Array.fill(numDatapoints)(Random.nextInt(numClasses)),
      features = Array.fill(numDatapoints, numFeatures)(Random.nextDouble))

    val regr = new DistributedLogisticRegression(
      numMachines = numMachines,
      numClasses = numClasses,
      numFeatures = numFeatures,
      stepSize = stepSize,
      regularizationFactor = regularizationFactor)

    regr.train(data, numIterations)
  }

  /*
  import kantan.csv._
  import kantan.csv.ops._
  import DistributedLogisticRegression.IterationInfo

  implicit val iterInfoEncoder = RowEncoder.encoder(0, 1, 2) { (i: IterationInfo) =>
    (i.loss, i.iterTime.toMillis, i.totalTime.toMillis)
  }

  implicit val iterInfoDecoder = RowDecoder.decoder(0, 1, 2) { (loss: Double, iterTime: Long, totalTime: Long) =>
    IterationInfo(loss, java.time.Duration.ofMillis(iterTime), java.time.Duration.ofMillis(totalTime))
  }

  def saveIterationInfo(info: Seq[IterationInfo], filename: String): Unit = {
    val writer = new java.io.File(filename).asCsvWriter[IterationInfo](',', Seq("loss", "iter_time", "total_time"))
    writer.write(info).close()
  }

  def loadIterationInfo(filename: String): Seq[IterationInfo] =
    new java.io.File(filename).asUnsafeCsvReader[IterationInfo](',', header = true).toSeq
  */
}
