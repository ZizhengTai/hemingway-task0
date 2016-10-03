import scala.util.Random

import breeze.numerics.{log2, pow}
import breeze.plot._
import org.apache.spark.{SparkConf, SparkContext}
import org.jfree.chart.axis.NumberTickUnit

object Main {
  import hemingway._

  def main(args: Array[String]): Unit = {
    // Extract arguments
    val dataDir = args(0)
    val numClasses = args(1).toInt
    val stepSize = args(2).toDouble
    val numStepSizeIterations = args(3).toInt
    val regParam = args(4).toDouble
    val numIterations = args(5).toInt
    val numMachines = args(6).toInt
    val outputDir = args(7)

    // Create Spark context
    val conf = new SparkConf()
      .setAppName("hemingway-task0")
      //.setMaster("local[8]")
      .setMaster(s"local[$numMachines]")
    implicit val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    // Warm up Spark
    warmUp(numMachines)

    // Load MNIST dataset
    val data = new MnistLoader(dataDir)

    // Regression for each number of machines
    /*
    case class Result(
      stepSize: Double,
      numStepSizeIterations: Int,
      numIterationsToConverge: Int)

    val configs = for {
      stepSize <- Seq(1e-2, 8e-3)
      numStepSizeIterations <- Seq(15, 20, 25, 30)
    } yield (stepSize, numStepSizeIterations)
    */

    val results = Seq(1, 8, 32) map { m =>
      val regr = new DistributedLogisticRegression(
        numMachines = m,
        numClasses = numClasses,
        numFeatures = data.trainImages(0).length,
        initStepSize = stepSize,
        stepSize = { (s, i) => s / pow(2, log2(i.toDouble / numStepSizeIterations + 1)) },
        regParam = regParam)

      regr.train(
        data = data.trainingSet,
        numIterations = numIterations,
        stopLoss = 0.238)

      regr

      /*
      println(s">>> Finished: $s $n ${regr.iterationInfo.length}")
      new File(s"$outputDir/$s $n ${regr.iterationInfo.length}").createNewFile()

      Result(s, n, regr.iterationInfo.length)
      */
    }

    /*
    // Find configuration that converged fastest
    val best = results.minBy(_.numIterationsToConverge)
    println(s">>> Best configuration converged in ${best.numIterationsToConverge} iterations:")
    println(s">>> stepSize = ${best.stepSize}, numStepSizeIterations = ${best.numStepSizeIterations}")
    new File(s"$outputDir/Best ${best.stepSize} ${best.numStepSizeIterations} ${best.numIterationsToConverge}")
      .createNewFile()
    */

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
    //ps(0).logScaleY = true
    ps(0).yaxis.setTickUnit(new NumberTickUnit(lossDiff / 10))
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

    for ((r, m) <- results zip Seq(1, 8, 32)) {
      val iters = r.iterationInfo.indices map (1.0 + _)
      val losses = r.iterationInfo map (_.loss)
      val iterTime = r.iterationInfo map (_.iterTime.toMillis.toDouble)
      val totalTime = r.iterationInfo map (_.totalTime.toMillis.toDouble)

      ps(0) += plot(iters, losses, name = m.toString)
      ps(1) += plot(iters, iterTime, name = m.toString)
      ps(2) += plot(totalTime, losses, name = m.toString)
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
    val initStepSize = 1e-3
    val regParam = 1e-5
    val numIterations = 3

    val data =
      (Array.fill(numDatapoints)(Random.nextInt(numClasses)),
       Array.fill(numDatapoints, numFeatures)(Random.nextDouble)).zipped map LabeledPoint

    val regr = new DistributedLogisticRegression(
      numMachines = numMachines,
      numClasses = numClasses,
      numFeatures = numFeatures,
      initStepSize = initStepSize,
      regParam = regParam)

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
